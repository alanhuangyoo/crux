import { describe, expect, it } from "vitest";
import { applyEditsToNormalizedContent } from "../src/core/tools/edit-diff.ts";

/**
 * "The old text must match exactly" is true and says nothing about what the
 * file contains, so a failed edit is a dead end: the model guesses again or
 * spends a turn re-reading. Measured over 89 trials, edits failed 43 times in
 * 528 calls and 23 were this error, while `read` was called 156 times against
 * those 528 edits.
 */

const FILE = [
	"def parse(value):",
	"    cleaned = value.strip()",
	"    if not cleaned:",
	"        return None",
	"    return int(cleaned)",
].join("\n");

function failureFor(oldText: string): string {
	try {
		applyEditsToNormalizedContent(FILE, [{ oldText, newText: "x" }], "/app/parse.py");
	} catch (error) {
		return (error as Error).message;
	}
	return "";
}

describe("an edit that does not match", () => {
	it("shows where in the file it was probably aiming", () => {
		// Stale: the file says `cleaned`, the edit remembers `trimmed`.
		const message = failureFor("    trimmed = value.strip()");
		expect(message).toContain("closest thing in the file is around line 2");
		expect(message).toContain("cleaned = value.strip()");
		expect(message).toMatch(/\s2\t/);
	});

	it("says to re-read when the file may have moved on", () => {
		expect(failureFor("    trimmed = value.strip()")).toContain("read the file again");
	});

	it("falls back to the plain message when nothing in the file is close", () => {
		const message = failureFor("import tensorflow as tf");
		expect(message).toContain("must match exactly");
		expect(message).not.toContain("closest thing");
	});

	it("still succeeds when the text is there", () => {
		const result = applyEditsToNormalizedContent(
			FILE,
			[{ oldText: "        return None", newText: "        return 0" }],
			"/app/parse.py",
		);
		expect(result.newContent).toContain("return 0");
	});
});
