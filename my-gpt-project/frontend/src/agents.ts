import {
  Agent,
  MCPServerSSE,
  connectMcpServers,
  getAllMcpTools,
  tool,
} from '@openai/agents';
import z from 'zod';

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
        
        console.log(`[MCP] Active MCP servers: ${mcpServers.active.length}`);
        console.log(`[MCP] Failed MCP servers: ${mcpServers.failed.length}`);
        for (const [server, error] of mcpServers.errors) {
          console.warn(`[MCP] ${server.name} failed to connect: ${error.message}`);
        }

        // 验证每个服务器的工具
        for (const server of mcpServers.active) {
          try {
            const tools = await server.listTools();
            console.log(`[MCP] Server "${server.name}" has ${tools.length} tools:`);
            tools.forEach((tool) => {
              console.log(`  - ${tool.name}: ${tool.description || 'No description'}`);
            });
          } catch (error) {
            console.error(`[MCP] Error listing tools from ${server.name}:`, error);
          }
        }

        // 验证 MCP 工具是否加载成功
        if (mcpServers.active.length > 0) {
          try {
            const mcpTools = await getAllMcpTools(mcpServers.active);
            console.log(`[MCP] Total loaded ${mcpTools.length} tools from all MCP servers:`);
            mcpTools.forEach((tool) => {
              console.log(`  - ${tool.name}: ${tool.type === 'function' ? tool.description : 'MCP tool'}`);
            });
          } catch (error) {
            console.error('[MCP] Error loading MCP tools:', error);
          }
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
          instructions: 'You are a basic agent with Python execution and web browsing capabilities.',
          model: 'o4-mini',
          tools: allTools,
          mcpServers: mcpServers.active,
        });

        // 验证 Agent 的所有工具（包括 MCP 工具）
        try {
          // 创建一个临时的 runContext 来获取工具
          const { RunContext } = await import('@openai/agents');
          const tempContext = new RunContext({});
          const allTools = await agent.getAllTools(tempContext);
          console.log(`[Agent] Agent has ${allTools.length} total tools:`);
          allTools.forEach((tool) => {
            if (tool.type === 'function') {
              console.log(`  - ${tool.name} (function): ${tool.description}`);
            } else {
              console.log(`  - ${tool.name} (${tool.type})`);
            }
          });
        } catch (error) {
          console.error('[Agent] Error getting all tools:', error);
        }

        return agent;
      })();
    }
    agentInstance = await agentPromise;
  }
  return agentInstance;
}
