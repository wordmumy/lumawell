// src/App.tsx —— 全屏 + 头像(message.user.avatar) + “正在输入…” + 不丢失用户消息（正确使用 updateMsg）
import React, { useEffect, useMemo, useRef, useState } from 'react'
import Chat, { useMessages, Bubble } from '@chatui/core'
import '@chatui/core/dist/index.css'

type ChatMessage = {
  _id: string
  type: 'text' | 'typing' | string
  content: any
  position?: 'left' | 'right'
  user?: { avatar?: string }   // ChatUI v3 运行时会读取 user.avatar
}

const uid = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

// 放两张图到 public/：/public/bot.png、/public/user.png（没有也能跑，只是头像空）
const BOT_AVATAR = '/bot.png'
const USER_AVATAR = '/user.png'

// 把纯文本拆成多气泡，并写入 user.avatar
function toBubbles(text: string, avatar: string, position: 'left' | 'right' = 'left') {
  return String(text || '')
    .split(/\n\n+/)
    .map(t => t.trim())
    .filter(Boolean)
    .map((t) => ({
      _id: uid(),
      type: 'text',
      content: { text: t },
      user: { avatar },         // ★ 头像在 user.avatar
      position,
    })) as any[]
}

function getThreadId() {
  const k = 'chat_thread_id'
  let v = localStorage.getItem(k)
  if (!v) { v = uid(); localStorage.setItem(k, v) }
  return v
}

export default function App() {
  // 外层容器控制全屏
  const containerStyle = useMemo(() => ({
    height: '100dvh',
    width: '100vw',
    maxWidth: '100vw',
    overflow: 'hidden',
    background: 'var(--c-bg, #f7f8fa)',
  }), [])

  // ★ 正确拿到 updateMsg（注意它需要两个参数）
  const { messages, appendMsg, updateMsg, resetList } = useMessages([])
  const threadId = getThreadId()

  // “正在输入 …” 占位
  const typingIdRef = useRef<string | null>(null)
  const [thinking, setThinking] = useState(false)

  // 欢迎语
  useEffect(() => {
    appendMsg({
      _id: uid(),
      type: 'text',
      content: { text: '你好，我是你的 LumaWell Chatbot！' },
      position: 'left',
      user: { avatar: BOT_AVATAR },
    } as any)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const renderMessageContent = (msg: ChatMessage) => {
    if (msg.type === 'typing') return <Bubble><span className="typing-dots">对方正在输入</span></Bubble>
    if (msg.type === 'text')   return <Bubble>{(msg.content as any)?.text}</Bubble>
    return <Bubble>（暂不支持的消息类型）</Bubble>
  }

  async function askBackend(userText: string) {
    const url = '/api/chat'  // 走 Vite 代理

    // 1) 先追加“正在输入 …”占位
    setThinking(true)
    const tid = uid()
    typingIdRef.current = tid
    appendMsg({
      _id: tid,
      type: 'typing',
      content: {},
      position: 'left',
      user: { avatar: BOT_AVATAR },
    } as any)

    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, message: userText, city: null, realtime: true }),
      })

      if (!resp.ok) {
        const raw = await resp.text()
        let detail = raw
        try { detail = JSON.parse(raw).detail || detail } catch {}
        throw new Error(`${resp.status} ${detail || '(empty body)'}`)
      }

      const data = await resp.json()
      const answer = data?.reply ?? ''
      const chunks = toBubbles(answer, BOT_AVATAR, 'left')

      // 2) ★ 不删除消息，直接把 typing 替换成第一段内容 —— 正确的 updateMsg(id, msg)
      if (chunks.length) {
        const [first, ...rest] = chunks
        // 第一段替换 typing（不要带 _id）
        const { _id: _omit, ...firstWithoutId } = first as any
        updateMsg(tid, firstWithoutId as any)

        // 其余段落顺次追加
        rest.forEach((m, i) => {
          const { _id: _omit2, ...msgWithoutId } = (m as any)
          setTimeout(() => appendMsg({ _id: uid(), ...(msgWithoutId as any) }), i * 60)
        })
      } else {
        updateMsg(tid, {
          type: 'text',
          content: { text: '（后端无内容）' },
          position: 'left',
          user: { avatar: BOT_AVATAR },
        } as any)
      }

    } catch (e: any) {
      // 3) ★ 出错也不删消息，直接把 typing 改成错误文本 —— 仍然是 updateMsg(id, msg)
      updateMsg(tid, {
        type: 'text',
        content: { text: `请求失败：${e?.message || String(e)}` },
        position: 'left',
        user: { avatar: BOT_AVATAR },
      } as any)
    } finally {
      setThinking(false)
      typingIdRef.current = null
    }
  }

  function handleSend(type: string, val: any) {
    if (type !== 'text') return
    const text = String(val || '').trim()
    if (!text) return

    // 用户消息始终追加，且不会被删
    appendMsg({
      _id: uid(),
      type: 'text',
      content: { text },
      position: 'right',
      user: { avatar: USER_AVATAR },
    } as any)

    askBackend(text)
  }

  const quickReplies = [
    { name: '最近皮肤干燥怎么护肤' },
    { name: '今晚吃什么更健康' },
    { name: '悉尼这周天气' },
    { name: '如何改善睡眠' },
  ]
  const handleQuickReplyClick = (item: any) => item?.name && handleSend('text', item.name)

  return (
    // 把类名加在外层容器，避免给 <Chat> 传 className 触发 TS 报错
    <div className="lw-fullscreen lw-chat" style={containerStyle}>
      <Chat
        navbar={{ title: 'LumaWell Chat' }}
        messages={messages}
        renderMessageContent={renderMessageContent}
        quickReplies={quickReplies as any}
        onQuickReplyClick={handleQuickReplyClick}
        onSend={handleSend}
        placeholder="输入消息…"
      />

      <button
        onClick={() => resetList([])}
        className="lw-clear"
        disabled={thinking}
        title={thinking ? '对方正在输入，稍后再清空' : '清空对话'}
      >
        清空对话
      </button>
    </div>
  )
}
