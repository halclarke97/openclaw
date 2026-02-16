/**
 * Rolling context eviction extension.
 *
 * Intercepts the SDK's `session_before_compact` event and replaces the default
 * LLM-generated summary with a minimal eviction note.  The old messages are
 * still persisted in the session JSONL and remain searchable via memory_search;
 * we simply don't spend tokens summarizing them.
 */
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import {
  estimateTokens,
  findCutPoint,
  type ExtensionAPI,
  type FileOperations,
  type SessionEntry,
} from "@mariozechner/pi-coding-agent";

const KEEP_RATIO = 0.5;
const MIN_DROP_TOKENS = 1024;

type RollingCut = {
  firstKeptEntryId: string;
  firstKeptIndex: number;
  droppedTokens: number;
  droppedMessages: AgentMessage[];
};

function createFileOps(): FileOperations {
  return {
    read: new Set(),
    written: new Set(),
    edited: new Set(),
  };
}

function isMessageEntry(entry: SessionEntry): entry is Extract<SessionEntry, { type: "message" }> {
  return entry.type === "message";
}

function estimateEntryTokens(entry: SessionEntry): number {
  if (!isMessageEntry(entry)) {
    return 0;
  }
  return estimateTokens(entry.message);
}

function estimateBoundaryTokens(entries: SessionEntry[], start: number, end: number): number {
  let tokens = 0;
  for (let i = start; i < end; i++) {
    tokens += estimateEntryTokens(entries[i]);
  }
  return tokens;
}

function collectDroppedMessages(
  entries: SessionEntry[],
  start: number,
  end: number,
): AgentMessage[] {
  const messages: AgentMessage[] = [];
  for (let i = start; i < end; i++) {
    const entry = entries[i];
    if (isMessageEntry(entry)) {
      messages.push(entry.message);
    }
  }
  return messages;
}

function findBoundaryStartIndex(entries: SessionEntry[]): number {
  for (let i = entries.length - 1; i >= 0; i--) {
    if (entries[i].type === "compaction") {
      return i + 1;
    }
  }
  return 0;
}

function resolveRollingCut(entries: SessionEntry[]): RollingCut | null {
  if (entries.length === 0) {
    return null;
  }

  const boundaryStart = findBoundaryStartIndex(entries);
  const boundaryEnd = entries.length;
  if (boundaryStart >= boundaryEnd) {
    return null;
  }

  const boundaryTokens = estimateBoundaryTokens(entries, boundaryStart, boundaryEnd);
  if (boundaryTokens <= 0) {
    return null;
  }

  const keepRecentTokens = Math.max(1, Math.floor(boundaryTokens * KEEP_RATIO));
  const cutPoint = findCutPoint(entries, boundaryStart, boundaryEnd, keepRecentTokens);
  let firstKeptIndex = cutPoint.firstKeptEntryIndex;

  // Rolling mode does not summarize split-turn prefixes, so keep full turns.
  if (cutPoint.isSplitTurn && cutPoint.turnStartIndex >= boundaryStart) {
    firstKeptIndex = cutPoint.turnStartIndex;
  }

  if (firstKeptIndex <= boundaryStart || firstKeptIndex >= boundaryEnd) {
    return null;
  }

  const firstKeptEntryId = entries[firstKeptIndex]?.id;
  if (typeof firstKeptEntryId !== "string" || firstKeptEntryId.length === 0) {
    return null;
  }

  const droppedTokens = estimateBoundaryTokens(entries, boundaryStart, firstKeptIndex);
  if (droppedTokens < MIN_DROP_TOKENS) {
    return null;
  }

  const droppedMessages = collectDroppedMessages(entries, boundaryStart, firstKeptIndex);
  if (droppedMessages.length === 0) {
    return null;
  }

  return {
    firstKeptEntryId,
    firstKeptIndex,
    droppedTokens,
    droppedMessages,
  };
}

function extractFileOpsFromMessage(message: AgentMessage, fileOps: FileOperations): void {
  if (message.role !== "assistant" || !Array.isArray(message.content)) {
    return;
  }
  for (const block of message.content) {
    if (!block || typeof block !== "object" || block.type !== "toolCall") {
      continue;
    }
    const args = block.arguments as Record<string, unknown>;
    const path = typeof args.path === "string" ? args.path : undefined;
    if (!path) {
      continue;
    }
    if (block.name === "read") {
      fileOps.read.add(path);
    } else if (block.name === "write") {
      fileOps.written.add(path);
    } else if (block.name === "edit") {
      fileOps.edited.add(path);
    }
  }
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
    const rollingCut = resolveRollingCut(event.branchEntries);
    if (!rollingCut) {
      return { cancel: true };
    }

    const fileOps = createFileOps();
    for (const message of rollingCut.droppedMessages) {
      extractFileOpsFromMessage(message, fileOps);
    }
    const { readFiles, modifiedFiles } = computeFileLists(fileOps);
    const fileOpsSummary = formatFileOperations(readFiles, modifiedFiles);

    const note =
      `[Rolling eviction: ${rollingCut.droppedMessages.length} messages dropped (~${rollingCut.droppedTokens} tokens). ` +
      `Old messages remain in session JSONL and are searchable via memory_search. ` +
      `Use memory_search to recall prior conversation content.]` +
      fileOpsSummary;

    return {
      compaction: {
        summary: note,
        firstKeptEntryId: rollingCut.firstKeptEntryId,
        tokensBefore: event.preparation.tokensBefore,
        details: { readFiles, modifiedFiles },
      },
    };
  });
}

export const __testing = {
  KEEP_RATIO,
  MIN_DROP_TOKENS,
  estimateEntryTokens,
  estimateBoundaryTokens,
  findBoundaryStartIndex,
  resolveRollingCut,
  extractFileOpsFromMessage,
  computeFileLists,
  formatFileOperations,
} as const;
