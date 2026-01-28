'use client';

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ExternalLink, Copy, Check } from 'lucide-react';
import { cn } from '@/lib/utils';
import { FadeIn } from '@/components/ui/LoadingEffects';

type TextMessageProps = {
  text: string;
  isUser: boolean;
};

/**
 * 代码块组件（用于 Markdown 渲染）
 */
function MarkdownCodeBlock({
  inline,
  className,
  children,
  ...props
}: {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}) {
  const [copied, setCopied] = React.useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';
  const codeString = String(children).replace(/\n$/, '');

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // 内联代码
  if (inline) {
    return (
      <code
        className="px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-800 text-sm font-mono"
        {...props}
      >
        {children}
      </code>
    );
  }

  // 代码块
  return (
    <div className="relative group my-4 rounded-lg overflow-hidden">
      {/* 工具栏 */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-800 border-b border-zinc-700">
        <span className="text-xs text-zinc-400">
          {language || 'code'}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200 transition-colors opacity-0 group-hover:opacity-100"
        >
          {copied ? (
            <>
              <Check size={12} className="text-green-400" />
              <span className="text-green-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: 0,
          fontSize: '13px',
        }}
      >
        {codeString}
      </SyntaxHighlighter>
    </div>
  );
}

/**
 * 自定义链接组件
 */
function MarkdownLink({
  href,
  children,
}: {
  href?: string;
  children?: React.ReactNode;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 hover:underline transition-colors"
    >
      {children}
      <ExternalLink size={12} className="inline-block" />
    </a>
  );
}

/**
 * Markdown 渲染组件
 */
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // 代码块
        code: MarkdownCodeBlock as any,
        // 链接
        a: MarkdownLink as any,
        // 段落
        p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
        // 标题
        h1: ({ children }) => (
          <h1 className="text-2xl font-bold mt-6 mb-3">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-xl font-bold mt-5 mb-2">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-lg font-semibold mt-4 mb-2">{children}</h3>
        ),
        h4: ({ children }) => (
          <h4 className="text-base font-semibold mt-3 mb-1">{children}</h4>
        ),
        // 列表
        ul: ({ children }) => (
          <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>
        ),
        li: ({ children }) => <li className="text-zinc-700">{children}</li>,
        // 引用
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-zinc-200 pl-4 italic text-zinc-600 my-3">
            {children}
          </blockquote>
        ),
        // 水平线
        hr: () => <hr className="my-4 border-zinc-200" />,
        // 表格
        table: ({ children }) => (
          <div className="overflow-x-auto my-4">
            <table className="min-w-full border-collapse border border-zinc-200 rounded-lg overflow-hidden">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-zinc-50">{children}</thead>
        ),
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => (
          <tr className="border-b border-zinc-200 last:border-b-0">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="px-4 py-2 text-left text-sm font-semibold text-zinc-700 border-r border-zinc-200 last:border-r-0">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-4 py-2 text-sm text-zinc-600 border-r border-zinc-200 last:border-r-0">
            {children}
          </td>
        ),
        // 删除线
        del: ({ children }) => (
          <del className="text-zinc-400 line-through">{children}</del>
        ),
        // 强调
        strong: ({ children }) => (
          <strong className="font-semibold text-zinc-900">{children}</strong>
        ),
        em: ({ children }) => <em className="italic">{children}</em>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

/**
 * 文本消息组件
 */
export function TextMessage({ text, isUser }: TextMessageProps) {
  return (
    <FadeIn>
      <div
        className={cn('flex flex-row gap-2', {
          'justify-end py-2': isUser,
        })}
      >
        <div
          className={cn('rounded-2xl', {
            // 用户消息样式
            'px-4 py-2.5 max-w-[70%] ml-4 text-zinc-900 bg-zinc-100': isUser,
            // AI 消息样式 - 使用 Markdown 渲染
            'px-4 py-3 max-w-[85%] mr-4 text-zinc-800 bg-white': !isUser,
          })}
        >
          {isUser ? (
            // 用户消息直接显示文本
            <span className="text-[15px] leading-relaxed">{text}</span>
          ) : (
            // AI 消息使用 Markdown 渲染
            <div className="prose prose-sm prose-zinc max-w-none text-[15px] leading-relaxed">
              <MarkdownContent content={text} />
            </div>
          )}
        </div>
      </div>
    </FadeIn>
  );
}
