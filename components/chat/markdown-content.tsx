"use client"

import React, { useState } from "react"
import ReactMarkdown, { type Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeHighlight from "rehype-highlight"

interface MarkdownContentProps {
  content: string
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  // Basic guard
  if (!content) return null

  interface CodeRendererProps {
    inline?: boolean
    className?: string
    children?: React.ReactNode[] | React.ReactNode
    [key: string]: any
  }

  const codeRenderer = ({ inline, className, children, ...props }: CodeRendererProps) => {
      const language = (className || "").match(/language-(\w+)/)?.[1]
      if (inline) {
        return (
          <code className="bg-slate-800 px-1 py-0.5 rounded text-[0.85em]" {...props}>
            {children}
          </code>
        )
      }
      return (
        <CodeBlock language={language} className={className} {...props}>
          {children as React.ReactNode}
        </CodeBlock>
      )
  }
  const components: Components = {
    code: codeRenderer,
    a({ node, children, ...props }) {
      return (
        <a
          className="text-blue-400 hover:text-blue-300 underline"
          target="_blank"
          rel="noopener noreferrer"
          {...props}
        >
          {children}
        </a>
      )
    },
    table({ node, children, ...props }) {
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse" {...props}>
            {children}
          </table>
        </div>
      )
    },
    th({ node, children, ...props }) {
      return (
        <th className="border border-slate-700 bg-slate-800 px-2 py-1 text-left font-medium" {...props}>
          {children}
        </th>
      )
    },
    td({ node, children, ...props }) {
      return (
        <td className="border border-slate-700 px-2 py-1 align-top" {...props}>
          {children}
        </td>
      )
    },
  }

  return (
    <ReactMarkdown
      className="prose prose-invert max-w-none prose-pre:rounded-md prose-pre:bg-slate-900 prose-code:text-blue-300 prose-headings:text-slate-100 prose-p:text-slate-200 prose-li:marker:text-slate-500"
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight as any]}
      components={components}
    >
      {content}
    </ReactMarkdown>
  )
}

interface CodeBlockProps extends React.HTMLAttributes<HTMLElement> {
  language?: string
  children: React.ReactNode
}

function CodeBlock({ language, children, className, ...props }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const onCopy = () => {
    if (typeof children === "string") {
      navigator.clipboard.writeText(children).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1800)
      })
    } else if (Array.isArray(children)) {
      const text = children.join("")
      navigator.clipboard.writeText(text).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 1800)
      })
    }
  }

  return (
    <div className="relative group">
      {language && (
        <span className="absolute top-1 left-2 text-[10px] uppercase tracking-wide text-slate-400 bg-slate-800/70 px-1.5 py-0.5 rounded">
          {language}
        </span>
      )}
      <button
        type="button"
        onClick={onCopy}
        className="opacity-0 group-hover:opacity-100 transition-opacity absolute top-1 right-1 text-[10px] bg-slate-700 hover:bg-slate-600 text-slate-200 px-2 py-0.5 rounded"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <pre className="overflow-x-auto text-xs !mt-5" {...props}>
        <code className={className}>{children}</code>
      </pre>
    </div>
  )
}
