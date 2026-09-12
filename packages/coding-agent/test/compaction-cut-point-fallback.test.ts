import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { describe, expect, it } from "vitest";
import { findCutPoint } from "../src/core/compaction/compaction.ts";
import type { SessionEntry, SessionMessageEntry } from "../src/core/session-manager.ts";

/**
 * A cut point is a user or assistant message, never a tool result. So when the
 * keep budget is spent by trailing tool results alone, there is nothing at or
 * after that entry to cut at -- and the search used to fall through to
 * `cutPoints[0]`, which keeps the conversation from its beginning. Compaction
 * then ran, reported success, and removed nothing.
 *
 * One `read` of a large file is thousands of tokens against a keep budget that,
 * on a 32K window, is 4,096. Measured across one arm, 67% of compactions removed
 * nothing; this was part of it.
 */

let counter = 0;
let previous: string | null = null;

function entry(message: AgentMessage): SessionMessageEntry {
	const id = `e${counter++}`;
	const built: SessionMessageEntry = {
		type: "message",
		id,
		parentId: previous,
		timestamp: new Date().toISOString(),
		message,
	};
	previous = id;
	return built;
}

function user(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: Date.now() } as unknown as AgentMessage;
}

function assistantCalling(id: string): AgentMessage {
	return {
		role: "assistant",
		content: [{ type: "toolCall", id, name: "read", arguments: {} }],
		usage: {
			input: 1,
			output: 1,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 2,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: Date.now(),
		api: "openai-completions",
		provider: "p",
		model: "m",
	} as unknown as AgentMessage;
}

function hugeResult(id: string, tokens: number): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: "read",
		content: [{ type: "text", text: "x".repeat(tokens * 4) }],
		details: {},
		isError: false,
		timestamp: Date.now(),
	} as unknown as AgentMessage;
}

function session(): SessionEntry[] {
	counter = 0;
	previous = null;
	return [
		entry(user("the task")),
		entry(assistantCalling("c1")),
		entry(hugeResult("c1", 3000)),
		entry(user("carry on")),
		entry(assistantCalling("c2")),
		entry(hugeResult("c2", 9000)),
	];
}

describe("a keep budget spent entirely by a trailing tool result", () => {
	it("cuts at the last point there is instead of keeping everything", () => {
		const entries = session();
		const result = findCutPoint(entries, 0, entries.length, 4096);

		// Anything but 0: keeping from the first entry is keeping the whole
		// conversation, which is what the bug did.
		expect(result.firstKeptEntryIndex).toBeGreaterThan(0);
		expect(result.firstKeptEntryIndex).toBe(4);
	});

	it("still honours the budget when a cut point does sit inside it", () => {
		const entries = session();
		const result = findCutPoint(entries, 0, entries.length, 13000);
		expect(result.firstKeptEntryIndex).toBeLessThan(4);
	});

	it("keeps everything only when there is genuinely nowhere to cut", () => {
		counter = 0;
		previous = null;
		const entries: SessionEntry[] = [entry(hugeResult("c0", 9000))];
		expect(findCutPoint(entries, 0, entries.length, 4096).firstKeptEntryIndex).toBe(0);
	});
});
