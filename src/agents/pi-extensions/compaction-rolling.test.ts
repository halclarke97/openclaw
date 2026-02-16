import type { ExtensionAPI, SessionEntry } from "@mariozechner/pi-coding-agent";
import { describe, expect, it } from "vitest";
import compactionRollingExtension, { __testing } from "./compaction-rolling.js";

function createHandler(): (event: unknown) => Promise<unknown> {
  let handler: ((event: unknown) => Promise<unknown>) | undefined;
  const api = {
    on(event: string, callback: (payload: unknown) => Promise<unknown>) {
      if (event === "session_before_compact") {
        handler = callback;
      }
    },
  } as unknown as ExtensionAPI;
  compactionRollingExtension(api);
  if (!handler) {
    throw new Error("session_before_compact handler was not registered");
  }
  return handler;
}

function compactionEntry(id: string, parentId: string | null): SessionEntry {
  return {
    type: "compaction",
    id,
    parentId,
    timestamp: new Date().toISOString(),
    summary: "prior summary",
    firstKeptEntryId: id,
    tokensBefore: 100,
  } as SessionEntry;
}

function userEntry(id: string, parentId: string | null, chars: number): SessionEntry {
  return {
    type: "message",
    id,
    parentId,
    timestamp: new Date().toISOString(),
    message: {
      role: "user",
      content: "x".repeat(chars),
      timestamp: Date.now(),
    },
  } as SessionEntry;
}

function assistantWithToolCallsEntry(
  id: string,
  parentId: string | null,
  textChars: number,
): SessionEntry {
  return {
    type: "message",
    id,
    parentId,
    timestamp: new Date().toISOString(),
    message: {
      role: "assistant",
      content: [
        { type: "text", text: "y".repeat(textChars) },
        { type: "toolCall", id: "call-read-a", name: "read", arguments: { path: "/tmp/a.ts" } },
        { type: "toolCall", id: "call-write-b", name: "write", arguments: { path: "/tmp/b.ts" } },
        { type: "toolCall", id: "call-edit-a", name: "edit", arguments: { path: "/tmp/a.ts" } },
        { type: "toolCall", id: "call-read-c", name: "read", arguments: { path: "/tmp/c.ts" } },
      ],
      timestamp: Date.now(),
    },
  } as SessionEntry;
}

describe("compaction-rolling cut and token logic", () => {
  it("uses SDK chars/4 token estimation for message entries", () => {
    const entry = userEntry("u1", null, 9);
    expect(__testing.estimateEntryTokens(entry)).toBe(3);
  });

  it("finds boundary after the last compaction and computes a rolling cut", () => {
    const entries: SessionEntry[] = [
      compactionEntry("c0", null),
      userEntry("u1", "c0", 5000),
      userEntry("u2", "u1", 5000),
      userEntry("u3", "u2", 5000),
    ];

    expect(__testing.findBoundaryStartIndex(entries)).toBe(1);
    const cut = __testing.resolveRollingCut(entries);
    expect(cut?.firstKeptEntryId).toBe("u2");
    expect(cut?.droppedTokens).toBe(1250);
    expect(cut?.droppedMessages).toHaveLength(1);
  });

  it("returns null when there are no message tokens to cut", () => {
    const entries: SessionEntry[] = [
      compactionEntry("c0", null),
      {
        type: "model_change",
        id: "m1",
        parentId: "c0",
        timestamp: new Date().toISOString(),
        provider: "openai",
        modelId: "gpt-5",
      } as SessionEntry,
      {
        type: "label",
        id: "l1",
        parentId: "m1",
        timestamp: new Date().toISOString(),
        targetId: "m1",
        label: "checkpoint",
      } as SessionEntry,
    ];

    expect(__testing.resolveRollingCut(entries)).toBeNull();
  });
});

describe("compaction-rolling extension hook", () => {
  it("cancels when dropping less than MIN_DROP_TOKENS", async () => {
    const handler = createHandler();
    const entries: SessionEntry[] = [
      compactionEntry("c0", null),
      userEntry("u1", "c0", 1000),
      userEntry("u2", "u1", 5000),
      userEntry("u3", "u2", 5000),
    ];

    const result = (await handler({
      preparation: { tokensBefore: 99999 },
      branchEntries: entries,
    })) as { cancel?: boolean };

    expect(result.cancel).toBe(true);
  });

  it("cancels when boundary is too close to the start", async () => {
    const handler = createHandler();
    const entries: SessionEntry[] = [compactionEntry("c0", null), userEntry("u1", "c0", 5000)];

    const result = (await handler({
      preparation: { tokensBefore: 77777 },
      branchEntries: entries,
    })) as { cancel?: boolean };

    expect(result.cancel).toBe(true);
  });

  it("returns rolling compaction with extracted file operations", async () => {
    const handler = createHandler();
    const entries: SessionEntry[] = [
      compactionEntry("c0", null),
      assistantWithToolCallsEntry("a1", "c0", 5000),
      userEntry("u2", "a1", 5000),
      userEntry("u3", "u2", 5000),
    ];

    const result = (await handler({
      preparation: { tokensBefore: 222222 },
      branchEntries: entries,
    })) as {
      compaction?: {
        summary: string;
        firstKeptEntryId: string;
        tokensBefore: number;
        details?: { readFiles?: string[]; modifiedFiles?: string[] };
      };
    };

    expect(result.compaction?.firstKeptEntryId).toBe("u2");
    expect(result.compaction?.tokensBefore).toBe(222222);
    expect(result.compaction?.details?.readFiles).toEqual(["/tmp/c.ts"]);
    expect(result.compaction?.details?.modifiedFiles).toEqual(["/tmp/a.ts", "/tmp/b.ts"]);
    expect(result.compaction?.summary).toContain("Rolling eviction: 1 messages dropped");
    expect(result.compaction?.summary).toContain("memory_search");
    expect(result.compaction?.summary).toContain("<read-files>");
    expect(result.compaction?.summary).toContain("<modified-files>");
  });
});

describe("compaction-rolling constants", () => {
  it("keeps roughly half the boundary context and requires a meaningful drop", () => {
    expect(__testing.KEEP_RATIO).toBe(0.5);
    expect(__testing.MIN_DROP_TOKENS).toBe(1024);
  });
});
