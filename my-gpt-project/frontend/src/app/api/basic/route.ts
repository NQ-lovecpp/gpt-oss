import { NextRequest } from 'next/server';
import { randomUUID } from 'node:crypto';

import { getAgent } from '@/agents';
import { Runner, RunState, RunToolApprovalItem } from '@openai/agents';
import type { AgentInputItem } from '@openai/agents';
import { db } from '@/db';

function generateConversationId() {
  return `conv_${randomUUID().replace(/-/g, '').slice(0, 24)}`;
}

// SSE 编码器
function encodeSSE(event: string, data: unknown) {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export async function POST(req: NextRequest) {
  try {
    const data = await req.json();
    let { messages, conversationId, decisions, stream: useStream } = data;

    if (!messages) {
      messages = [];
    }

    if (!conversationId) {
      conversationId = generateConversationId();
    }

    if (!decisions) {
      decisions = null;
    }

    const agent = await getAgent();
    const runner = new Runner({
      groupId: conversationId,
    });

    // 如果请求流式输出
    if (useStream) {
      const encoder = new TextEncoder();
      
      const readable = new ReadableStream({
        async start(controller) {
          try {
            // 发送 conversationId
            controller.enqueue(encoder.encode(encodeSSE('init', { conversationId })));

            let input: AgentInputItem[] | RunState<any, any>;
            
            if (Object.keys(decisions || {}).length > 0 && data.conversationId) {
              const stateString = await db().get(data.conversationId);
              if (!stateString) {
                controller.enqueue(encoder.encode(encodeSSE('error', { error: 'Conversation not found' })));
                controller.close();
                return;
              }
              const state = await RunState.fromString(agent, stateString);
              const interruptions = state.getInterruptions();
              interruptions.forEach((item: RunToolApprovalItem) => {
                if (item.type === 'tool_approval_item' && 'callId' in item.rawItem) {
                  const callId = item.rawItem.callId;
                  if (decisions[callId] === 'approved') {
                    state.approve(item);
                  } else if (decisions[callId] === 'rejected') {
                    state.reject(item);
                  }
                }
              });
              input = state;
            } else {
              input = messages;
            }

            // 使用流式运行
            const stream = await runner.run(agent, input, { 
              stream: true,
            });

            // 处理流事件
            for await (const event of stream) {
              if (event.type === 'raw_model_stream_event') {
                // 发送原始模型事件（用于打字机效果）
                const eventData = event.data as any;
                if (eventData?.type === 'response.output_text.delta' || eventData?.type === 'output_text_delta') {
                  controller.enqueue(encoder.encode(encodeSSE('text_delta', { 
                    delta: eventData.delta 
                  })));
                }
              } else if (event.type === 'run_item_stream_event') {
                // 发送运行项目事件
                const item = event.item as any;
                if (item.type === 'tool_call_item') {
                  controller.enqueue(encoder.encode(encodeSSE('tool_call', {
                    name: item.rawItem.name,
                    arguments: item.rawItem.arguments,
                    callId: item.rawItem.callId,
                    status: 'in_progress'
                  })));
                } else if (item.type === 'tool_call_output_item') {
                  controller.enqueue(encoder.encode(encodeSSE('tool_output', {
                    callId: item.rawItem.callId,
                    output: item.output,
                    status: 'completed'
                  })));
                } else if (item.type === 'message_output_item') {
                  controller.enqueue(encoder.encode(encodeSSE('message', {
                    role: 'assistant',
                    content: item.content
                  })));
                }
              } else if (event.type === 'agent_updated_stream_event') {
                controller.enqueue(encoder.encode(encodeSSE('agent_update', {
                  agent: (event as any).agent?.name || 'Unknown'
                })));
              }
            }

            // 等待流完成
            await stream.completed;

            // 检查是否有中断（需要审批）
            if (stream.interruptions && stream.interruptions.length > 0) {
              await db().set(conversationId, JSON.stringify(stream.state));
              controller.enqueue(encoder.encode(encodeSSE('interruption', {
                approvals: stream.interruptions
                  .filter((item) => item.type === 'tool_approval_item')
                  .map((item) => item.toJSON()),
                history: stream.history
              })));
            } else {
              // 发送完成事件
              controller.enqueue(encoder.encode(encodeSSE('done', {
                response: stream.finalOutput,
                history: stream.history
              })));
            }

            controller.close();
          } catch (error) {
            console.error('Stream error:', error);
            controller.enqueue(encoder.encode(encodeSSE('error', { 
              error: error instanceof Error ? error.message : 'Unknown error' 
            })));
            controller.close();
          }
        },
      });

      return new Response(readable, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }

    // 非流式模式（保持原有逻辑）
    let input: AgentInputItem[] | RunState<any, any>;
    if (Object.keys(decisions || {}).length > 0 && data.conversationId) {
      const stateString = await db().get(data.conversationId);
      if (!stateString) {
        return Response.json({ error: 'Conversation not found' }, { status: 404 });
      }
      const state = await RunState.fromString(agent, stateString);
      const interruptions = state.getInterruptions();
      interruptions.forEach((item: RunToolApprovalItem) => {
        if (item.type === 'tool_approval_item' && 'callId' in item.rawItem) {
          const callId = item.rawItem.callId;
          if (decisions[callId] === 'approved') {
            state.approve(item);
          } else if (decisions[callId] === 'rejected') {
            state.reject(item);
          }
        }
      });
      input = state;
    } else {
      input = messages;
    }

    const result = await runner.run(agent, input);

    if (result.interruptions.length > 0) {
      await db().set(conversationId, JSON.stringify(result.state));
      return Response.json({
        conversationId,
        approvals: result.interruptions
          .filter((item) => item.type === 'tool_approval_item')
          .map((item) => item.toJSON()),
        history: result.history,
      });
    }

    return Response.json({
      response: result.finalOutput,
      history: result.history,
      conversationId,
    });
  } catch (error) {
    console.error(error);
    return Response.json({ error: 'Internal server error' }, { status: 500 });
  }
}
