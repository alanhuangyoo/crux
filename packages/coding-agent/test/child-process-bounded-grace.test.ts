import { spawn } from "node:child_process";
import { describe, expect, it } from "vitest";
import { waitForChildProcess } from "../src/utils/child-process.ts";

/**
 * The post-exit grace exists so the tail of a command's output is not truncated,
 * and it re-arms on every chunk. A detached descendant that keeps writing to the
 * inherited pipe re-arms it forever.
 *
 * Found live: `nohup python3 ocr5.py > log 2>&1 &` left a trial silent for 318
 * minutes after a tool call that had asked for a 120-second timeout. The tool's
 * timeout only kills `child.pid`, which by then had exited, while the wait it was
 * meant to end stayed pending -- so the code that checks for the timeout was
 * never reached.
 */

/** A shell that exits immediately, leaving a child holding its stdout. */
function shellWithChattyOrphan() {
	return spawn("sh", ["-c", "( while true; do echo tick; sleep 0.05; done ) & exit 7"], {
		stdio: ["ignore", "pipe", "pipe"],
	});
}

describe("a command whose orphan keeps talking", () => {
	it("stops waiting instead of being held open forever", async () => {
		const child = shellWithChattyOrphan();
		child.stdout?.on("data", () => {});
		child.stderr?.on("data", () => {});
		const started = Date.now();

		const code = await waitForChildProcess(child);

		const elapsed = Date.now() - started;
		expect(elapsed).toBeLessThan(6_000);
		expect(code).toBe(7);
		if (child.pid) {
			try {
				process.kill(-child.pid, "SIGKILL");
			} catch {
				// the shell is already gone; the orphan is reparented and harmless here
			}
		}
	}, 20_000);

	it("still waits for a quiet command's own exit code", async () => {
		const child = spawn("sh", ["-c", "echo done; exit 3"], { stdio: ["ignore", "pipe", "pipe"] });
		let seen = "";
		child.stdout?.on("data", (chunk) => {
			seen += String(chunk);
		});
		expect(await waitForChildProcess(child)).toBe(3);
		expect(seen.trim()).toBe("done");
	}, 20_000);
});
