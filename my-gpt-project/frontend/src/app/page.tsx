'use client';

import type { AgentInputItem, RunToolApprovalItem } from '@openai/agents';
import { useState, useCallback, useRef } from 'react';
import { App } from '@/components/App';
import { Approvals } from '@/components/Approvals';

// 流式事件类型
type StreamEvent = 
  | { type: 'init'; data: { conversationId: string } }
  | { type: 'text_delta'; data: { delta: string } }
  | { type: 'tool_call'; data: { name: string; arguments: string; callId: string; status: string } }
  | { type: 'tool_output'; data: { callId: string; output: string; status: string } }
  | { type: 'message'; data: { role: string; content: string } }
  | { type: 'agent_update'; data: { agent: string } }
  | { type: 'interruption'; data: { approvals: any[]; history: AgentInputItem[] } }
  | { type: 'done'; data: { response: string; history: AgentInputItem[] } }
  | { type: 'error'; data: { error: string } };

export default function Home() {
  const [history, setHistory] = useState<AgentInputItem[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [approvals, setApprovals] = useState<
    ReturnType<RunToolApprovalItem['toJSON']>[]
  >([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  
  // 用于跟踪当前的工具调用
  const pendingToolCalls = useRef<Map<string, any>>(new Map());

  const processStreamEvent = useCallback((event: StreamEvent) => {
    switch (event.type) {
      case 'init':
        setConversationId(event.data.conversationId);
        break;

      case 'text_delta':
        // 累积文本增量（打字机效果）
        setStreamingText(prev => prev + event.data.delta);
        break;

      case 'tool_call':
        // 添加工具调用到历史
        pendingToolCalls.current.set(event.data.callId, event.data);
        setHistory(prev => {
          // 检查是否已经存在这个工具调用
          const existing = prev.find(
            item => item.type === 'function_call' && (item as any).callId === event.data.callId
          );
          if (existing) return prev;
          
          // 过滤掉占位消息
          const filtered = prev.filter(item => {
            if (item.type === 'message' && item.role === 'assistant') {
              const content = item.content;
              return !(Array.isArray(content) && content.length === 0);
            }
            return true;
          });
          
          return [
            ...filtered,
            {
              type: 'function_call',
              name: event.data.name,
              arguments: event.data.arguments,
              callId: event.data.callId,
              id: event.data.callId,
            } as unknown as AgentInputItem
          ];
        });
        break;

      case 'tool_output':
        // 更新工具调用结果
        setHistory(prev => {
          const newHistory = [...prev];
          // 添加函数调用结果
          newHistory.push({
            type: 'function_call_result',
            callId: event.data.callId,
            output: event.data.output,
          } as AgentInputItem);
          return newHistory;
        });
        pendingToolCalls.current.delete(event.data.callId);
        break;

      case 'message':
        // 添加消息到历史
        setStreamingText('');
        break;

      case 'agent_update':
        // 可以用来显示当前活动的 agent
        console.log('Agent update:', event.data.agent);
        break;

      case 'interruption':
        // 需要审批
        setHistory(event.data.history);
        setApprovals(event.data.approvals);
        setIsStreaming(false);
        setStreamingText('');
        break;

      case 'done':
        // 完成
        setHistory(event.data.history);
        setApprovals([]);
        setIsStreaming(false);
        setStreamingText('');
        break;

      case 'error':
        console.error('Stream error:', event.data.error);
        setIsStreaming(false);
        setStreamingText('');
        break;
    }
  }, []);

  async function makeStreamingRequest({
    message,
    decisions,
  }: {
    message?: string;
    decisions?: Map<string, 'approved' | 'rejected'>;
  }) {
    const messages = [...history];

    if (message) {
      messages.push({ type: 'message', role: 'user', content: message });
      setHistory(messages);
    }

    setIsStreaming(true);
    setStreamingText('');
    pendingToolCalls.current.clear();

    try {
      const response = await fetch('/api/basic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages,
          conversationId,
          decisions: Object.fromEntries(decisions ?? []),
          stream: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No reader available');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        // 解析 SSE 事件
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 保留不完整的行

        let eventType = '';
        let eventData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            eventType = line.slice(7);
          } else if (line.startsWith('data: ')) {
            eventData = line.slice(6);
          } else if (line === '' && eventType && eventData) {
            // 完整事件
            try {
              const data = JSON.parse(eventData);
              processStreamEvent({ type: eventType, data } as StreamEvent);
            } catch (e) {
              console.error('Failed to parse event data:', e);
            }
            eventType = '';
            eventData = '';
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      setIsStreaming(false);
      setStreamingText('');
    }
  }

  const handleSend = async (message: string) => {
    await makeStreamingRequest({ message });
  };

  async function handleDone(decisions: Map<string, 'approved' | 'rejected'>) {
    await makeStreamingRequest({ decisions });
  }

  // 构建显示用的历史（包含流式文本）
  const displayHistory = [...history];
  if (isStreaming && streamingText) {
    // 如果正在流式输出且有文本，添加临时消息
    displayHistory.push({
      type: 'message',
      role: 'assistant',
      content: [{ type: 'output_text', text: streamingText }],
      status: 'in_progress',
    } as unknown as AgentInputItem);
  } else if (isStreaming && !streamingText && pendingToolCalls.current.size === 0) {
    // 正在等待但还没有内容
    displayHistory.push({
      type: 'message',
      role: 'assistant',
      content: [],
      status: 'in_progress',
    } as unknown as AgentInputItem);
  }

  return (
    <>
      <App 
        history={displayHistory} 
        onSend={handleSend} 
        isStreaming={isStreaming}
      />
      <Approvals approvals={approvals} onDone={handleDone} />
    </>
  );
}
