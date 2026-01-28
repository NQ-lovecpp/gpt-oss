import {
  Agent,
  MCPServerSSE,
  connectMcpServers,
  getAllMcpTools,
  tool,
} from '@openai/agents';
import z from 'zod';
import OpenAI from 'openai';
import { OpenAIResponsesModel } from '@openai/agents';

const getWeather = tool({
  name: 'getWeather',
  description: 'Get the weather for a given city',
  parameters: z.object({
    city: z.string(),
  }),
  execute: async ({ city }) => {
    return `The weather in ${city} is sunny.`;
  },

  needsApproval: true,
});

// Python execution MCP server (SSE 模式，连接已运行的服务器)
const pythonServer = new MCPServerSSE({
  name: 'Python Execution Server',
  url: 'http://127.0.0.1:8000/sse',
});

// Web browser MCP server (基于 Exa，SSE 模式)
const browserServer = new MCPServerSSE({
  name: 'Web Browser Server (Exa)',
  url: 'http://127.0.0.1:8001/sse',
});

// 连接 MCP 服务器的单例
let mcpServersPromise: Promise<Awaited<ReturnType<typeof connectMcpServers>>> | null = null;

async function initializeMcpServers() {
  if (!mcpServersPromise) {
    mcpServersPromise = connectMcpServers([pythonServer, browserServer], {
      connectInParallel: true,
    });
  }
  return mcpServersPromise;
}

// 创建一个工厂函数来获取 agent
let agentInstance: Agent | null = null;
let agentPromise: Promise<Agent> | null = null;

export async function getAgent(): Promise<Agent> {
  if (!agentInstance) {
    if (!agentPromise) {
      agentPromise = (async () => {
        const mcpServers = await initializeMcpServers();

        // 简要日志
        if (mcpServers.failed.length > 0) {
          console.warn(`[MCP] ${mcpServers.failed.length} server(s) failed to connect`);
        }

        // 拉取 MCP 工具并与本地工具合并
        const baseTools = [getWeather];
        let mcpTools = [];
        try {
          mcpTools = mcpServers.active.length
            ? await getAllMcpTools(mcpServers.active)
            : [];
        } catch (error) {
          console.error('[MCP] Error loading MCP tools (will continue with base tools):', error);
        }
        const allTools = [...baseTools, ...mcpTools];

        const agent = new Agent({
          name: 'Basic Agent',
          instructions: `You are a helpful assistant with Python execution and web browsing capabilities.

IMPORTANT: You MUST always think and reason in English, regardless of what language the user uses. Your internal reasoning process should always be in English. However, you should respond to the user in their language.

For example:
- If user asks in Chinese: Think in English internally, then respond in Chinese
- If user asks in English: Think in English, respond in English
- If user asks in any other language: Think in English, respond in that language

This rule about reasoning in English is mandatory and must never be violated.`,
          model: 'o4-mini',
          modelSettings: {
            reasoning: { effort: 'medium', summary: 'detailed' },
            text: { verbosity: 'medium' },
          },
          tools: allTools,
          mcpServers: mcpServers.active,
        });

        console.log(`[Agent] Initialized with ${allTools.length} tools`);
        return agent;
      })();
    }
    agentInstance = await agentPromise;
  }
  return agentInstance;
}
