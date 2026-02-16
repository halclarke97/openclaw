/**
 * Rolling context eviction extension.
 *
 * Intercepts the SDK's `session_before_compact` event and replaces the default
 * LLM-generated summary with a simple eviction note. Instead of using the SDK's
 * aggressive cut point (~20k tokens kept), this extension computes its own cut
 * point that keeps ~50% of context, dropping only the oldest messages.
 *
 * The old messages are still persisted in the session JSONL and remain
 * searchable via memory_search; we simply don't spend tokens summarizing them.
 */
import type { ExtensionAPI, FileOperations } from "@mariozechner/pi-coding-agent";

/** Target: keep this fraction of tokensBefore after compaction */
const KEEP_RATIO = 0.5;

/** Minimum tokens to drop — don't bother compacting if we'd drop less than this */
const MIN_DROP_TOKENS = 5000;

interface SessionEntry {
  type: string;
  id: string;
  message?: {
    role: string;
    content?: unknown;
  };
  summary?: string;
}

/**
 * Rough token estimate matching the SDK's estimateTokens logic (chars/4).
 */
function estimateEntryTokens(entry: SessionEntry): number {
  if (entry.type !== "message" || !entry.message) {
    return 0;
  }
  const msg = entry.message;
  let chars = 0;

  if (msg.role === "user") {
    const content = msg.content;
    if (typeof content === "string") {
      chars = content.length;
    } else if (Array.isArray(content)) {
      for (const block of content as Array<{ type: string; text?: string }>) {
        if (block.type === "text" && block.text) {
          chars += block.text.length;
        }
      }
    }
  } else if (msg.role === "assistant") {
    const content = (msg as any).content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (block.type === "text") {
          chars += (block.text || "").length;
        } else if (block.type === "thinking") {
          chars += (block.thinking || "").length;
        } else if (block.type === "toolCall") {
          chars += (block.name || "").length + JSON.stringify(block.arguments || {}).length;
        }
      }
    }
  } else if (msg.role === "toolResult") {
    const content = (msg as any).content;
    if (typeof content === "string") {
      chars = content.length;
    } else if (Array.isArray(content)) {
      for (const block of content as Array<{ type: string; text?: string }>) {
        if (block.type === "text" && block.text) {
          chars += block.text.length;
        }
      }
    }
  }

  return Math.ceil(chars / 4);
}

/**
 * Check if an entry is a valid cut point (same logic as SDK's findValidCutPoints).
 * We can cut at user/assistant messages but NOT at toolResult (would orphan it).
 */
function isValidCutPoint(entry: SessionEntry): boolean {
  if (entry.type === "branch_summary" || entry.type === "custom_message") {
    return true;
  }
  if (entry.type !== "message" || !entry.message) {
    return false;
  }
  const role = entry.message.role;
  return (
    role === "user" ||
    role === "assistant" ||
    role === "bashExecution" ||
    role === "custom" ||
    role === "branchSummary" ||
    role === "compactionSummary"
  );
}

function computeFileLists(fileOps: FileOperations): {
  readFiles: string[];
  modifiedFiles: string[];
} {
  const modified = new Set([...fileOps.edited, ...fileOps.written]);
  const readFiles = [...fileOps.read].filter((f) => !modified.has(f)).toSorted();
  const modifiedFiles = [...modified].toSorted();
  return { readFiles, modifiedFiles };
}

function formatFileOperations(readFiles: string[], modifiedFiles: string[]): string {
  const sections: string[] = [];
  if (readFiles.length > 0) {
    sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
  }
  if (modifiedFiles.length > 0) {
    sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
  }
  return sections.length > 0 ? `\n\n${sections.join("\n\n")}` : "";
}

export default function compactionRollingExtension(api: ExtensionAPI): void {
  api.on("session_before_compact", async (event) => {
    const { preparation, branchEntries } = event;
    const { readFiles, modifiedFiles } = computeFileLists(preparation.fileOps);
    const fileOpsSummary = formatFileOperations(readFiles, modifiedFiles);
    const tokensBefore = preparation.tokensBefore;

    // Find the boundary: start after the last compaction entry
    let boundaryStart = 0;
    for (let i = branchEntries.length - 1; i >= 0; i--) {
      if (branchEntries[i].type === "compaction") {
        boundaryStart = i + 1;
        break;
      }
    }

    // Calculate how many tokens to keep
    const keepTokens = Math.floor(tokensBefore * KEEP_RATIO);
    const dropTarget = tokensBefore - keepTokens;

    if (dropTarget < MIN_DROP_TOKENS) {
      // Not worth compacting — cancel
      return { cancel: true };
    }

    // Walk backwards from newest, accumulating tokens to find our cut point.
    // We want to KEEP ~keepTokens of the newest messages.
    let accumulatedTokens = 0;
    let cutIndex = boundaryStart; // Default: drop everything (fallback)
    let foundCut = false;

    for (let i = branchEntries.length - 1; i >= boundaryStart; i--) {
      const entry = branchEntries[i] as SessionEntry;
      const entryTokens = estimateEntryTokens(entry);
      accumulatedTokens += entryTokens;

      if (accumulatedTokens >= keepTokens) {
        // We've accumulated enough to keep. Find nearest valid cut point at or after i.
        for (let c = i; c < branchEntries.length; c++) {
          if (isValidCutPoint(branchEntries[c] as SessionEntry)) {
            cutIndex = c;
            foundCut = true;
            break;
          }
        }
        break;
      }
    }

    // If we couldn't find a good cut point, or we'd keep everything, use SDK's plan
    if (!foundCut || cutIndex <= boundaryStart) {
      // Fall through to SDK's default cut point
      const evictedCount = preparation.messagesToSummarize.length;
      const note =
        `[Rolling eviction: ${evictedCount} messages dropped from context. ` +
        `Old messages remain in session JSONL and are searchable via memory_search. ` +
        `Use memory_search to recall prior conversation content.]` +
        fileOpsSummary;

      return {
        compaction: {
          summary: note,
          firstKeptEntryId: preparation.firstKeptEntryId,
          tokensBefore: preparation.tokensBefore,
          details: { readFiles, modifiedFiles },
        },
      };
    }

    const firstKeptEntry = branchEntries[cutIndex] as SessionEntry;
    const firstKeptEntryId = firstKeptEntry.id;

    // Count how many message entries we're dropping
    let droppedMessages = 0;
    for (let i = boundaryStart; i < cutIndex; i++) {
      if ((branchEntries[i] as SessionEntry).type === "message") {
        droppedMessages++;
      }
    }

    // Estimate tokens being dropped
    let droppedTokens = 0;
    for (let i = boundaryStart; i < cutIndex; i++) {
      droppedTokens += estimateEntryTokens(branchEntries[i] as SessionEntry);
    }

    const note =
      `[Rolling eviction: ${droppedMessages} messages dropped from context. ` +
      `Old messages remain in session JSONL and are searchable via memory_search. ` +
      `Use memory_search to recall prior conversation content.]` +
      fileOpsSummary;

    return {
      compaction: {
        summary: note,
        firstKeptEntryId,
        tokensBefore,
        details: {
          readFiles,
          modifiedFiles,
          droppedMessages,
          droppedTokens,
          keptTokensEstimate: tokensBefore - droppedTokens,
        },
      },
    };
  });
}
