import type { AssistantMessage } from "@earendil-works/pi-ai/compat";
import { describe, expect, it } from "vitest";
import { fitSummarizationRequest, summaryText } from "../src/core/compaction/compaction.ts";

/**
 * Compaction summarises the history it is dropping, and the summary it asks for
 * comes out of the same window that history is already filling. Nothing checked
 * the sum. Across 89 trials, 2,115 of 2,370 compactions failed -- 400s for
 * asking 16,409 + 16,384 against a 32,768 window, and length-stops for a budget
 * too small to finish in -- and each failure left the history untouched, so the
 * context only grew.
 */

function reply(stopReason: AssistantMessage["stopReason"], text: string): AssistantMessage {
	return { role: "assistant", content: [{ type: "text", text }], stopReason } as unknown as AssistantMessage;
}

const WINDOW = 32768;

describe("a summarization request that fits", () => {
	it("leaves a request that already fits alone", () => {
		const prompt = "x".repeat(4 * 8000);
		expect(fitSummarizationRequest(prompt, WINDOW, 4096)).toEqual({ promptText: prompt, maxTokens: 4096 });
	});

	it("never asks for more than the window holds, at any history size", () => {
		for (const promptTokens of [1000, 8000, 16409, 24000, 30534, 32000, 40000, 120000]) {
			const prompt = "x".repeat(4 * promptTokens);
			const fitted = fitSummarizationRequest(prompt, WINDOW, 16384);
			expect(Math.ceil(fitted.promptText.length / 4) + fitted.maxTokens).toBeLessThanOrEqual(WINDOW);
			expect(fitted.maxTokens).toBeGreaterThanOrEqual(512);
		}
	});

	it("spends the prompt, not the summary, once there is no room left", () => {
		const fitted = fitSummarizationRequest("x".repeat(4 * 32000), WINDOW, 16384);
		expect(fitted.promptText.length).toBeLessThan(4 * 32000);
		expect(fitted.promptText).toContain("middle of the conversation omitted");
	});

	it("keeps both ends when it drops the middle", () => {
		const prompt = `THE-TASK${"m".repeat(4 * 40000)}WHAT-JUST-HAPPENED`;
		const fitted = fitSummarizationRequest(prompt, WINDOW, 16384);
		expect(fitted.promptText.startsWith("THE-TASK")).toBe(true);
		expect(fitted.promptText.endsWith("WHAT-JUST-HAPPENED")).toBe(true);
	});

	it("does not meddle when the window is large or unknown", () => {
		const prompt = "x".repeat(4 * 30000);
		expect(fitSummarizationRequest(prompt, 200000, 16384).maxTokens).toBe(16384);
		expect(fitSummarizationRequest(prompt, 0, 16384).promptText).toBe(prompt);
	});
});

describe("a summary that stopped early", () => {
	it("is kept, because the alternative is no compaction at all", () => {
		const text = summaryText(reply("length", "## Work Done\n" + "a real summary. ".repeat(40)), "Summarization");
		expect(text).toContain("a real summary.");
		expect(text).toContain("cut off at the token cap");
	});

	it("is discarded when it is only a fragment", () => {
		expect(() => summaryText(reply("length", "## Work"), "Summarization")).toThrow(/token cap/);
	});

	it("is untouched on a clean stop", () => {
		expect(summaryText(reply("stop", "done"), "Summarization")).toBe("done");
	});

	it("still throws on a real error", () => {
		const bad = { ...reply("error", ""), errorMessage: "400 status code (no body)" } as AssistantMessage;
		expect(() => summaryText(bad, "Summarization")).toThrow(/400/);
	});
});
