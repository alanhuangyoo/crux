import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model } from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { generateSummaryWithUsage } from "../src/core/compaction/index.ts";

/**
 * Fitting a summarization request by a character ratio cannot be made correct,
 * because the ratio belongs to the text: measured against this deployment's own
 * tokenizer it runs from 3.99 on prose to 1.95 on what `write-compressor`
 * accumulates. A constant that fits one overflows the other, which is how a
 * fitted request still came back
 *
 *     400 ... Requested token count exceeds the model's maximum context length
 *
 * on 45 of one arm's compactions. A rejection is information, so the request
 * halves and goes again instead of throwing the summary away.
 */

const { completeSimpleMock } = vi.hoisted(() => ({ completeSimpleMock: vi.fn() }));

vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@earendil-works/pi-ai/compat")>();
	return { ...actual, completeSimple: completeSimpleMock };
});

const usage = {
	input: 10,
	output: 10,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 20,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function reply(over: Partial<AssistantMessage>): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text: "## Goal\nA complete summary of the work so far." }],
		api: "openai-completions",
		provider: "p",
		model: "m",
		usage,
		stopReason: "stop",
		timestamp: Date.now(),
		...over,
	} as AssistantMessage;
}

function model(): Model<"openai-completions"> {
	return {
		id: "m",
		name: "M",
		api: "openai-completions",
		provider: "p",
		baseUrl: "http://127.0.0.1:1",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 32768,
		maxTokens: 8192,
	} as unknown as Model<"openai-completions">;
}

/** Dense enough that a ratio-based fit overflows: 1.95 characters per token. */
function denseHistory(): AgentMessage[] {
	return [{ role: "user", content: "q".repeat(400_000), timestamp: 1 } as unknown as AgentMessage];
}

function promptLengthOf(call: unknown[]): number {
	const context = call[1] as { messages: Array<{ content: Array<{ text: string }> }> };
	return context.messages[0].content[0].text.length;
}

async function summarize() {
	return generateSummaryWithUsage(denseHistory(), model(), 16384, undefined);
}

describe("a summarization request the server rejects", () => {
	beforeEach(() => completeSimpleMock.mockReset());

	it("halves the prompt and goes again rather than losing the compaction", async () => {
		completeSimpleMock
			.mockResolvedValueOnce(reply({ stopReason: "error", errorMessage: "400 status code (no body)" }))
			.mockResolvedValueOnce(reply({}));

		const result = await summarize();

		expect(result.text).toContain("A complete summary");
		expect(completeSimpleMock).toHaveBeenCalledTimes(2);
		const [first, second] = completeSimpleMock.mock.calls;
		expect(promptLengthOf(second)).toBeLessThan(promptLengthOf(first) / 1.9);
	});

	it("keeps both ends of what it retries with", async () => {
		completeSimpleMock
			.mockResolvedValueOnce(reply({ stopReason: "error", errorMessage: "400" }))
			.mockResolvedValueOnce(reply({}));
		await summarize();
		const retried = completeSimpleMock.mock.calls[1];
		const context = retried[1] as { messages: Array<{ content: Array<{ text: string }> }> };
		expect(context.messages[0].content[0].text).toContain("middle of the conversation omitted");
	});

	it("stops after three attempts instead of retrying forever", async () => {
		completeSimpleMock.mockResolvedValue(reply({ stopReason: "error", errorMessage: "400 status code (no body)" }));
		await expect(summarize()).rejects.toThrow(/400/);
		expect(completeSimpleMock).toHaveBeenCalledTimes(3);
	});

	it("does not retry a summary that merely stopped early but said enough", async () => {
		completeSimpleMock.mockResolvedValueOnce(
			reply({ stopReason: "length", content: [{ type: "text", text: `## Goal\n${"real content. ".repeat(40)}` }] }),
		);
		const result = await summarize();
		expect(result.text).toContain("cut off at the token cap");
		expect(completeSimpleMock).toHaveBeenCalledTimes(1);
	});
});
