import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import nodePath from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createVerifyToolDefinition } from "../../src/core/tools/verify.ts";

/**
 * The tool exists because of a failure mode a prompt cannot reach: on
 * Terminal-Bench 2.1, 19 of 24 failures were trials that ran to completion and
 * answered wrongly. The agent believed it was done. So the property under test
 * is not "it runs commands" -- it is that a claim cannot be closed by assertion,
 * and that finishing re-runs everything.
 */
describe("verify tool", () => {
	let dir: string;

	beforeEach(async () => {
		dir = await mkdtemp(nodePath.join(tmpdir(), "verify-"));
	});
	afterEach(async () => {
		await rm(dir, { recursive: true, force: true });
	});

	const tool = () => createVerifyToolDefinition(dir);
	const call = (t: ReturnType<typeof tool>, args: any) => t.execute("id", args, undefined, undefined, { cwd: dir } as any);

	it("binds a claim to a command and reports its real exit status", async () => {
		const t = tool();
		const pass = await call(t, { action: "check", claim: "true holds", command: "true" });
		expect(pass.details?.failing).toBe(0);
		const fail = await call(t, { action: "check", claim: "false holds", command: "false" });
		expect(fail.details?.failing).toBe(1);
		// A failing check stays on the list rather than being dropped.
		expect(fail.details?.items.map((i) => i.claim)).toContain("false holds");
	});

	it("refuses to submit when nothing is bound", async () => {
		const t = tool();
		const r = await call(t, { action: "submit" });
		expect(r.isError).toBe(true);
		// The point of the refusal: passing on an empty list would mean nothing.
		expect(String(r.content[0].text)).toContain("nothing to re-run");
	});

	it("catches a claim that held earlier and was broken since", async () => {
		// The single most common way a task fails while looking finished: a
		// requirement met at step 10 and quietly undone at step 40.
		const marker = nodePath.join(dir, "ok.txt");
		await writeFile(marker, "x");
		const t = tool();
		const bound = await call(t, { action: "check", claim: "marker exists", command: `test -f ${marker}` });
		expect(bound.details?.failing).toBe(0);

		await rm(marker);

		const submitted = await call(t, { action: "submit" });
		expect(submitted.isError).toBe(true);
		expect(submitted.details?.failing).toBe(1);
		expect(String(submitted.content[0].text)).toContain("FAIL on re-run");
	});

	it("passing submit still says what it does not cover", async () => {
		const t = tool();
		await call(t, { action: "check", claim: "true holds", command: "true" });
		const r = await call(t, { action: "submit" });
		expect(r.isError).toBeFalsy();
		// Measured: graders ran about six times as many checks as the agent bound,
		// and every failure was inside that gap. A green list must not read as done.
		expect(String(r.content[0].text)).toContain("does not say they cover");
	});

	it("rebinding a claim replaces it rather than duplicating", async () => {
		const t = tool();
		await call(t, { action: "check", claim: "same claim", command: "false" });
		const second = await call(t, { action: "check", claim: "same claim", command: "true" });
		expect(second.details?.items).toHaveLength(1);
		expect(second.details?.failing).toBe(0);
	});

	it("needs both halves of a check", async () => {
		const t = tool();
		const r = await call(t, { action: "check", claim: "no command given" });
		expect(r.isError).toBe(true);
	});
});
