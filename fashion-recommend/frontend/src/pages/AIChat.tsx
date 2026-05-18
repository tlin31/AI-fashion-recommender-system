import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Bot, User, Sparkles, CheckCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import apiService from '../services/api'

type TextMessage = {
  type: 'text'
  role: 'user' | 'assistant'
  content: string
}

type ApprovalMessage = {
  type: 'approval'
  traitUpdates: Record<string, any>[]
  threadId: string
}

type ApprovedChip = {
  type: 'approved'
}

type ChatMessage = TextMessage | ApprovalMessage | ApprovedChip

export default function AIChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      type: 'text',
      role: 'assistant',
      content: "Hello! I'm Fashion Curator, your personal style AI. I can help you explore fashion trends, discover products, and put together outfits. What can I help you with today?"
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | undefined>(undefined)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const hasPendingApproval = messages.some(m => m.type === 'approval')

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading || hasPendingApproval) return

    const userMessage: ChatMessage = { type: 'text', role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const username = localStorage.getItem('username') || 'guest'

      // [Go backend — stateless single-shot chat, no Gorse integration]
      // const history = messages
      //   .filter((m): m is TextMessage => m.type === 'text')
      //   .map(m => ({ role: m.role, content: m.content }))
      // const sessionId = sessionStorage.getItem('ai_session_id') ?? undefined
      // const goResponse = await apiService.chatWithAI(input, history, sessionId, username)
      // const assistantMessage: ChatMessage = { type: 'text', role: 'assistant', content: goResponse.message }
      // setMessages(prev => [...prev, assistantMessage])

      const response = await apiService.agentChat(username, input, sessionId)

      if (response.session_id) setSessionId(response.session_id)

      const assistantMessage: ChatMessage = { type: 'text', role: 'assistant', content: response.message }
      setMessages(prev => [...prev, assistantMessage])

      if (response.pending_approval && response.pending_trait_updates?.length > 0) {
        const approvalCard: ChatMessage = {
          type: 'approval',
          traitUpdates: response.pending_trait_updates,
          threadId: response.session_id,
        }
        setMessages(prev => [...prev, approvalCard])
      }
    } catch (error) {
      console.error('Agent chat error:', error)
      const errorMessage: ChatMessage = {
        type: 'text',
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleApprove = async (approved: boolean, cardThreadId: string) => {
    setMessages(prev => prev.filter(m => m.type !== 'approval'))
    setLoading(true)

    try {
      await apiService.agentResume(cardThreadId, approved)  // cardThreadId = session_id
      if (approved) {
        setMessages(prev => [...prev, { type: 'approved' }])
      }
    } catch (error) {
      console.error('Agent resume error:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 h-[calc(100vh-4rem)]">
      <div className="mb-6 text-center">
        <div className="flex items-center justify-center mb-2">
          <Bot className="h-8 w-8 text-primary-600 mr-2" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary-600 to-pink-600 bg-clip-text text-transparent">
            Fashion Curator
          </h1>
        </div>
        <p className="text-gray-600">Chat with your AI stylist for personalised fashion advice</p>
      </div>

      <div className="bg-white rounded-2xl shadow-lg p-6 mb-4 h-[calc(100%-12rem)] overflow-y-auto">
        <div className="space-y-4">
          {messages.map((message, index) => {
            if (message.type === 'approved') {
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex justify-center"
                >
                  <div className="flex items-center space-x-2 bg-green-50 border border-green-200 rounded-full px-5 py-2 text-green-700 text-sm font-medium shadow-sm">
                    <CheckCircle className="h-4 w-4" />
                    <span>Preferences saved</span>
                  </div>
                </motion.div>
              )
            }

            if (message.type === 'approval') {
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-center"
                >
                  <div className="w-full max-w-md bg-white border-2 border-amber-400 rounded-2xl shadow-lg overflow-hidden">
                    {/* header bar */}
                    <div className="bg-gradient-to-r from-amber-400 to-orange-400 px-5 py-4">
                      <div className="flex items-center space-x-2 mb-0.5">
                        <Sparkles className="h-5 w-5 text-white" />
                        <p className="text-base font-bold text-white tracking-wide">
                          Save style preferences?
                        </p>
                      </div>
                      <p className="text-xs text-amber-100 pl-7">Based on your last message</p>
                    </div>

                    {/* trait rows — flatten {field, value, score} or nested {style_preferences: {minimalist: 0.9}} */}
                    <div className="px-5 py-4 space-y-2">
                      {message.traitUpdates.flatMap((update, i) => {
                        // Simple format: {field, value, score}
                        if (update.field != null) {
                          const score = update.score != null ? Math.round(update.score * 100) : null
                          return [(
                            <div key={i} className="flex items-center justify-between">
                              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider w-28 shrink-0">
                                {String(update.field).replace(/_/g, ' ')}
                              </span>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-semibold text-gray-800 capitalize">{String(update.value)}</span>
                                {score != null && <span className="text-xs text-white bg-amber-400 rounded-full px-2 py-0.5 font-medium">{score}%</span>}
                              </div>
                            </div>
                          )]
                        }
                        // Nested format: {style_preferences: {minimalist: 0.9}, price_sensitivity: "medium"}
                        return Object.entries(update).map(([key, val]) => {
                          const label = key.replace(/_/g, ' ').replace(/\bpreferences\b/g, '').trim()
                          const entries = val != null && typeof val === 'object' ? Object.entries(val as Record<string, number>) : [[String(val), null]] as [string, number | null][]
                          return entries.map(([v, score], j) => (
                            <div key={`${i}-${key}-${j}`} className="flex items-center justify-between">
                              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider w-28 shrink-0">{label}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-semibold text-gray-800 capitalize">{v}</span>
                                {score != null && <span className="text-xs text-white bg-amber-400 rounded-full px-2 py-0.5 font-medium">{Math.round(score * 100)}%</span>}
                              </div>
                            </div>
                          ))
                        }).flat()
                      })}
                    </div>

                    {/* actions */}
                    <div className="px-5 pb-5 flex items-center gap-4">
                      <button
                        onClick={() => handleApprove(true, message.threadId)}
                        className="flex-1 py-2.5 bg-gradient-to-r from-amber-400 to-orange-400 hover:from-amber-500 hover:to-orange-500 text-white font-semibold rounded-xl shadow transition-all active:scale-95"
                      >
                        Confirm
                      </button>
                      <button
                        onClick={() => handleApprove(false, message.threadId)}
                        className="text-sm text-gray-400 hover:text-gray-600 transition-colors py-2.5 px-3"
                      >
                        Dismiss
                      </button>
                    </div>
                  </div>
                </motion.div>
              )
            }

            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start space-x-2 max-w-[80%] ${message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.role === 'user'
                      ? 'bg-primary-600'
                      : 'bg-gradient-to-br from-purple-500 to-pink-500'
                  }`}>
                    {message.role === 'user' ? (
                      <User className="h-5 w-5 text-white" />
                    ) : (
                      <Sparkles className="h-5 w-5 text-white" />
                    )}
                  </div>
                  <div className={`rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}>
                    {message.role === 'user' ? (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    ) : (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                          strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                          ul: ({ children }) => <ul className="list-disc list-inside space-y-1 mb-2">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-2">{children}</ol>,
                          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                          h3: ({ children }) => <h3 className="font-semibold text-sm mt-3 mb-1">{children}</h3>,
                          code: ({ children }) => <code className="bg-gray-200 rounded px-1 text-xs font-mono">{children}</code>,
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    )}
                  </div>
                </div>
              </motion.div>
            )
          })}

          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="flex items-start space-x-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <Sparkles className="h-5 w-5 text-white" />
                </div>
                <div className="bg-gray-100 rounded-2xl px-4 py-3">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="bg-white rounded-2xl shadow-lg p-4">
        <div className="flex items-end space-x-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={hasPendingApproval ? 'Respond to the preference update above first…' : 'Ask me anything about fashion…'}
            className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-primary-500 focus:border-transparent max-h-32 disabled:bg-gray-50 disabled:text-gray-400"
            rows={1}
            disabled={loading || hasPendingApproval}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading || hasPendingApproval}
            className="bg-gradient-to-r from-primary-600 to-pink-600 text-white p-3 rounded-xl hover:from-primary-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="h-5 w-5" />
          </button>
        </div>
      </div>
    </div>
  )
}
