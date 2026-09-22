import { describe, expect, it } from "vitest";
import { OutputAccumulator } from "../src/core/tools/output-accumulator.ts";

// pi kept only the tail of a truncated command, which drops what a build or a
// test run prints first: the first compiler error, the top of a traceback.
// Claude Code keeps the start, Codex keeps both ends. The model's result now
// keeps a share of the budget for the first lines; a live view still wants
// only the latest ones.

function feed(lines: number, options: ConstructorParameters<typeof OutputAccumulator>[0] = {}) {
	const acc = new OutputAccumulator({ maxLines: 100, maxBytes: 1_000_000, headShare: 0.2, ...options });
	for (let i = 1; i <= lines; i++) acc.append(Buffer.from(`line-${i}\n`));
	acc.finish();
	return acc;
}

describe("a truncated output keeps its first lines for the model", () => {
	it("shows the first lines, what was left out, then the last", () => {
		const { content, truncation } = feed(1000).snapshot({ withHead: true });
		expect(content.startsWith("line-1\nline-2\n")).toBe(true);
		expect(content).toContain("line-20\n");
		expect(content).not.toContain("line-21\n");
		expect(content).toContain("[... 900 lines omitted ...]");
		// 100 lines of budget, 20 of them the head: the tail is the last 80.
		expect(content).not.toContain("line-920\n");
		expect(content).toContain("line-921\n");
		expect(content.trimEnd().endsWith("line-1000")).toBe(true);
		expect(truncation.headLines).toBe(20);
		// The same budget as before, split: the head is not extra.
		expect(truncation.outputLines).toBe(100);
		expect(truncation.totalLines).toBe(1000);
	});

	it("leaves the live view on the tail alone", () => {
		const { content, truncation } = feed(1000).snapshot();
		expect(content).not.toContain("line-1\n");
		expect(content).not.toContain("omitted");
		expect(truncation.headLines).toBeUndefined();
		expect(truncation.outputLines).toBe(100);
	});

	it("changes nothing when the output fits", () => {
		const { content, truncation } = feed(50).snapshot({ withHead: true });
		expect(truncation.truncated).toBe(false);
		expect(content).not.toContain("omitted");
		expect(content.split("\n").filter(Boolean)).toHaveLength(50);
	});

	it("keeps only the tail with no head share, as before", () => {
		const { content } = feed(1000, { headShare: 0 }).snapshot({ withHead: true });
		expect(content).not.toContain("line-1\n");
		expect(content).not.toContain("omitted");
	});

	it("respects the byte budget for the head as well as the line budget", () => {
		const acc = new OutputAccumulator({ maxLines: 10_000, maxBytes: 1000, headShare: 0.2 });
		for (let i = 1; i <= 500; i++) acc.append(Buffer.from(`row-${String(i).padStart(4, "0")}\n`));
		acc.finish();
		const { content, truncation } = acc.snapshot({ withHead: true });
		// 200 bytes of head at 9 bytes a line.
		expect(truncation.headLines).toBe(22);
		expect(content).toContain("row-0022\n");
		expect(content).not.toContain("row-0023\n");
		expect(Buffer.byteLength(content)).toBeLessThan(1100);
	});

	it("falls back to the tail when a single line is too long to keep whole", () => {
		const acc = new OutputAccumulator({ maxLines: 100, maxBytes: 1000, headShare: 0.2 });
		acc.append(Buffer.from("short\n"));
		acc.append(Buffer.from(`${"x".repeat(5000)}`));
		acc.finish();
		const { content, truncation } = acc.snapshot({ withHead: true });
		expect(truncation.lastLinePartial).toBe(true);
		expect(content).not.toContain("omitted");
	});

	it("keeps the head across chunk boundaries that split a line", () => {
		const acc = new OutputAccumulator({ maxLines: 10, maxBytes: 1_000_000, headShare: 0.2 });
		acc.append(Buffer.from("fir"));
		acc.append(Buffer.from("st\nsec"));
		acc.append(Buffer.from("ond\n"));
		for (let i = 3; i <= 40; i++) acc.append(Buffer.from(`l${i}\n`));
		acc.finish();
		const { content } = acc.snapshot({ withHead: true });
		expect(content.startsWith("first\nsecond\n")).toBe(true);
	});
});
