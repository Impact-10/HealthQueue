import React from 'react';

function ChatMessage({ from, text }) {
  return (
    <div className={`chat-message ${from === 'bot' ? 'bot' : 'user'}`}>
      <div className="message-text">{text}</div>
    </div>
  );
}

export default ChatMessage;
