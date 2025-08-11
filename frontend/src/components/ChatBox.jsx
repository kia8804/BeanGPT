import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { FaPaperPlane } from 'react-icons/fa';

function ChatBox({ messages, onSendMessage, isLoading }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-4 ${
                msg.role === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200'
              }`}
            >
              <ReactMarkdown>{msg.content}</ReactMarkdown>
              
              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 text-sm">
                  <details>
                    <summary className="cursor-pointer">📄 Sources</summary>
                    <div className="mt-2 space-y-1">
                      {msg.sources.map((source, i) => {
                        // Check if this is a web source (format: "Web-1: URL")
                        const isWebSource = source.startsWith('Web-');
                        
                        let citationLabel, sourceUrl, displayText;
                        
                        if (isWebSource) {
                          // Extract citation label and URL from "Web-1: URL" format
                          const match = source.match(/^(Web-\d+):\s*(.*)$/);
                          if (match) {
                            citationLabel = match[1];
                            sourceUrl = match[2];
                            displayText = sourceUrl;
                          } else {
                            // Fallback if format doesn't match
                            citationLabel = `Web-${i + 1}`;
                            sourceUrl = source;
                            displayText = source;
                          }
                        } else {
                          // Regular DOI source
                          citationLabel = `${i + 1}`;
                          sourceUrl = `https://doi.org/${source}`;
                          displayText = source;
                        }
                        
                        return (
                          <a
                            key={i}
                            href={sourceUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block text-blue-500 hover:underline"
                          >
                            [{citationLabel}] {displayText}
                          </a>
                        );
                      })}
                    </div>
                  </details>
                </div>
              )}

              {/* Genes */}
              {msg.genes && msg.genes.length > 0 && (
                <div className="mt-2 text-sm">
                  <details>
                    <summary className="cursor-pointer">🧬 Genes Mentioned</summary>
                    <div className="mt-2 space-y-2">
                      {msg.genes.map((gene, i) => (
                        <div key={i}>
                          <strong>{gene.name}</strong>
                          <ReactMarkdown>{gene.summary}</ReactMarkdown>
                        </div>
                      ))}
                    </div>
                  </details>
                </div>
              )}

              {/* Full Markdown Table */}
              {msg.fullMarkdownTable && (
                <div className="mt-2">
                  <details>
                    <summary className="cursor-pointer">📊 Show Full Table</summary>
                    <div className="mt-2 overflow-x-auto">
                      <ReactMarkdown>{msg.fullMarkdownTable}</ReactMarkdown>
                    </div>
                  </details>
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t dark:border-gray-700">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about dry bean research..."
            className="flex-1 p-2 border rounded-lg dark:bg-gray-700 dark:border-gray-600 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <FaPaperPlane />
          </button>
        </div>
      </form>
    </div>
  );
}

export default ChatBox; 