import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { Effort, getBundledModel } from "@oh-my-pi/pi-ai";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";
import * as autoThinkingClassifier from "../src/auto-thinking/classifier";
import { AUTO_THINKING } from "../src/thinking";

describe("auto routing", () => {
	let tempDir: TempDir;
	let session: AgentSession;
	let sessionSettings: Settings;
	let modelRegistry: ModelRegistry;
	const authStorages: AuthStorage[] = [];

	beforeEach(async () => {
		tempDir = TempDir.createSync("@pi-auto-routing-");
		const authStorage = await AuthStorage.create(path.join(tempDir.path(), "testauth.db"));
		authStorages.push(authStorage);
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		modelRegistry = new ModelRegistry(authStorage, path.join(tempDir.path(), "models.yml"));
		sessionSettings = Settings.isolated();
	});

	afterEach(async () => {
		vi.restoreAllMocks();
		if (session) {
			await session.dispose();
		}
		for (const authStorage of authStorages.splice(0)) {
			authStorage.close();
		}
		try {
			tempDir.removeSync();
		} catch {
			/* EBUSY on Windows — temp dir will be cleaned up later */
		}
	});

	function getModel(id: string) {
		const model = getBundledModel("anthropic", id);
		if (!model) throw new Error(`Expected anthropic model ${id} to exist`);
		return model;
	}

	async function createAutoRoutingSession() {
		const defaultModel = getModel("claude-sonnet-4-5");
		const slowModel = getModel("claude-sonnet-4-6");

		const agent = new Agent({
			initialState: {
				model: defaultModel,
				systemPrompt: ["Test"],
				tools: [],
				messages: [],
				thinkingLevel: undefined,
			},
		});

		sessionSettings.set("autoRouting", true);
		sessionSettings.setModelRole("default", `${defaultModel.provider}/${defaultModel.id}`);
		sessionSettings.setModelRole("slow", `${slowModel.provider}/${slowModel.id}`);
		sessionSettings.setModelRole("route", `${slowModel.provider}/${slowModel.id}`);
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: sessionSettings,
			modelRegistry,
			thinkingLevel: AUTO_THINKING,
		});

		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined);
	}

	it("upgrades after 3 consecutive >=High efforts", async () => {
		await createAutoRoutingSession();
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");
		const setModelTemporarySpy = vi.spyOn(session, "setModelTemporary");

		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Implement a thread-safe LRU cache");
		expect(setModelTemporarySpy).not.toHaveBeenCalled();

		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Refactor distributed lock service to use Raft");
		expect(setModelTemporarySpy).not.toHaveBeenCalled();

		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Design multi-tenant PostgreSQL migration system");
		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);
		expect(setModelTemporarySpy.mock.calls[0]?.[0]?.id).toBe("claude-sonnet-4-6");
	});

	it("resets counter on a <=Medium interruption", async () => {
		await createAutoRoutingSession();
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");
		const setModelTemporarySpy = vi.spyOn(session, "setModelTemporary");

		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard 1");
		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard 2");
		classifierSpy.mockResolvedValue(Effort.Medium);
		await session.prompt("Medium interruption");

		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard again");

		expect(setModelTemporarySpy).not.toHaveBeenCalled();
	});

	it("XHigh also counts toward upgrade", async () => {
		await createAutoRoutingSession();
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");
		const setModelTemporarySpy = vi.spyOn(session, "setModelTemporary");

		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard 1");
		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard 2");
		classifierSpy.mockResolvedValue(Effort.XHigh);
		await session.prompt("Very hard 3");

		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);
	});

	it("downgrades after 2 consecutive <=Medium turns while routed", async () => {
		await createAutoRoutingSession();
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");
		const setModelTemporarySpy = vi.spyOn(session, "setModelTemporary");

		for (let i = 0; i < 3; i++) {
			classifierSpy.mockResolvedValue(Effort.High);
			await session.prompt("Hard question");
		}
		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);

		// 1st Medium → accumulate, no downgrade yet
		classifierSpy.mockResolvedValue(Effort.Medium);
		await session.prompt("Simple follow-up 1");
		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);

		// 2nd Medium → downgrade
		classifierSpy.mockResolvedValue(Effort.Medium);
		await session.prompt("Simple follow-up 2");
		expect(setModelTemporarySpy).toHaveBeenCalledTimes(2);
		expect(setModelTemporarySpy.mock.calls[1]?.[0]?.id).toBe("claude-sonnet-4-5");
	});

	it("reset reverse counter on High and stays routed", async () => {
		await createAutoRoutingSession();
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");
		const setModelTemporarySpy = vi.spyOn(session, "setModelTemporary");

		for (let i = 0; i < 3; i++) {
			classifierSpy.mockResolvedValue(Effort.High);
			await session.prompt("Hard question");
		}
		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);

		// One Medium — reverse counter at 1
		classifierSpy.mockResolvedValue(Effort.Medium);
		await session.prompt("Medium 1");
		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);

		// A High resets reverse counter
		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard again");
		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);
	});

	it("resets counter on manual model switch", async () => {
		await createAutoRoutingSession();
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");

		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard 1");
		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard 2");

		const otherModel = getModel("claude-sonnet-4-6");
		await session.setModelTemporary(otherModel);
		await session.setModel(getModel("claude-sonnet-4-5"));

		const setModelTemporarySpy = vi.spyOn(session, "setModelTemporary");
		classifierSpy.mockResolvedValue(Effort.High);
		await session.prompt("Hard after switch");
		await session.prompt("Hard 2");
		await session.prompt("Hard 3");

		expect(setModelTemporarySpy).toHaveBeenCalledTimes(1);
	});

	it("routed model is affected by auto thinking independently", async () => {
		await createAutoRoutingSession();
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");

		for (let i = 0; i < 3; i++) {
			classifierSpy.mockResolvedValue(Effort.High);
			await session.prompt("Hard question");
		}

		expect(session.isAutoThinking).toBe(true);
		expect(session.thinkingLevel).toBe(Effort.High);

		// 2 consecutive Low → downgrade
		classifierSpy.mockResolvedValue(Effort.Low);
		await session.prompt("Easy question 1");
		classifierSpy.mockResolvedValue(Effort.Low);
		await session.prompt("Easy question 2");

		expect(session.model?.id).toBe("claude-sonnet-4-5");
	});

	it("does not trigger routing when setting is disabled", async () => {
		const defaultModel = getModel("claude-sonnet-4-5");

		const agent = new Agent({
			initialState: {
				model: defaultModel,
				systemPrompt: ["Test"],
				tools: [],
				messages: [],
				thinkingLevel: undefined,
			},
		});

		sessionSettings.set("autoRouting", false);
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(),
			settings: sessionSettings,
			modelRegistry,
			thinkingLevel: AUTO_THINKING,
		});
		vi.spyOn(session.agent, "prompt").mockResolvedValue(undefined);
		const classifierSpy = vi.spyOn(autoThinkingClassifier, "classifyDifficulty");
		const setModelTemporarySpy = vi.spyOn(session, "setModelTemporary");

		for (let i = 0; i < 3; i++) {
			classifierSpy.mockResolvedValue(Effort.High);
			await session.prompt("Hard question");
		}
		expect(setModelTemporarySpy).not.toHaveBeenCalled();
	});
});
