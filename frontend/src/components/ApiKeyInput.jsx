import React, { useState, useEffect } from 'react';
import { FaKey, FaEye, FaEyeSlash, FaCheck, FaExclamationTriangle, FaCog, FaShieldAlt, FaLock, FaPlus } from 'react-icons/fa';

const ApiKeyInput = ({ darkMode, onApiKeyChange }) => {
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isValid, setIsValid] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isSaved, setIsSaved] = useState(false);

  // Load API key from localStorage on component mount
  useEffect(() => {
    const savedApiKey = localStorage.getItem('openai_api_key');
    if (savedApiKey) {
      setApiKey(savedApiKey);
      setIsValid(validateApiKey(savedApiKey));
      setIsSaved(true);
      onApiKeyChange(savedApiKey);
    }
  }, [onApiKeyChange]);

  // Validate OpenAI API key format
  const validateApiKey = (key) => {
    return key.startsWith('sk-') && key.length >= 20;
  };

  // Handle API key input changes
  const handleApiKeyChange = (e) => {
    const value = e.target.value.trim();
    setApiKey(value);
    const valid = validateApiKey(value);
    setIsValid(valid);
    
    if (valid) {
      localStorage.setItem('openai_api_key', value);
      setIsSaved(true);
      onApiKeyChange(value);
    } else {
      setIsSaved(false);
      onApiKeyChange('');
    }
  };

  // Clear API key
  const clearApiKey = () => {
    setApiKey('');
    setIsValid(false);
    setIsSaved(false);
    setIsExpanded(false);
    localStorage.removeItem('openai_api_key');
    onApiKeyChange('');
  };

  // Toggle expanded view
  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="flex items-center space-x-4">
      {/* Compact Status/Input Toggle */}
      {isSaved && !isExpanded ? (
        // Saved state - compact indicator
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 rounded-full bg-gradient-to-r from-green-500 to-emerald-600 flex items-center justify-center">
              <FaShieldAlt className="text-white text-xs" />
            </div>
            <span className="text-sm font-medium text-green-600 dark:text-green-400">
              API Key Ready
            </span>
          </div>
          <button
            onClick={toggleExpanded}
            className={`p-2 rounded-lg transition-colors ${
              darkMode 
                ? 'hover:bg-slate-700 text-slate-400 hover:text-slate-300' 
                : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'
            }`}
            title="Manage API Key"
          >
            <FaCog className="text-sm" />
          </button>
        </div>
      ) : (
        // Need API key or editing state
        <div className="flex items-center space-x-3">
          {!isExpanded ? (
            // Collapsed - show add button
            <button
              onClick={toggleExpanded}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 transition-all shadow-sm"
            >
              <FaKey className="text-sm" />
              <span className="text-sm font-medium">Add API Key</span>
            </button>
          ) : (
            // Expanded - show input
            <div className="flex items-center space-x-3 bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-600 p-3 shadow-lg">
              <div className="flex items-center space-x-2">
                <FaKey className="text-blue-500 text-sm" />
                <div className="relative">
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    value={apiKey}
                    onChange={handleApiKeyChange}
                    placeholder="sk-..."
                    className={`w-64 px-3 py-1.5 text-sm border rounded-lg bg-transparent font-mono transition-all focus:outline-none focus:ring-2 ${
                      apiKey
                        ? isValid
                          ? 'border-green-500 focus:ring-green-500/30'
                          : 'border-red-500 focus:ring-red-500/30'
                        : 'border-gray-300 dark:border-slate-500 focus:ring-blue-500/30'
                    }`}
                  />
                  <div className="absolute inset-y-0 right-0 flex items-center space-x-1 pr-2">
                    <button
                      type="button"
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="p-1 rounded transition-colors hover:bg-gray-100 dark:hover:bg-slate-700"
                    >
                      {showApiKey ? <FaEyeSlash className="text-xs" /> : <FaEye className="text-xs" />}
                    </button>
                    {apiKey && (
                      <div className="ml-1">
                        {isValid ? (
                          <FaCheck className="text-green-500 text-xs" />
                        ) : (
                          <FaExclamationTriangle className="text-red-500 text-xs" />
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {isSaved && (
                <button
                  onClick={clearApiKey}
                  className="px-3 py-1.5 text-xs rounded-lg bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-900/30 transition-colors"
                >
                  Clear
                </button>
              )}
              
              <button
                onClick={toggleExpanded}
                className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-700 transition-colors"
              >
                <span className="text-xs">×</span>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Status indicator for no API key */}
      {!isSaved && !isExpanded && (
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse"></div>
          <span className="text-xs text-red-600 dark:text-red-400 font-medium">
            API Key Required
          </span>
        </div>
      )}
      
      {/* Get API Key Link - only show when expanded and no key */}
      {isExpanded && !isSaved && (
        <a 
          href="https://platform.openai.com/api-keys" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-xs text-blue-600 dark:text-blue-400 hover:underline whitespace-nowrap"
        >
          Get API Key →
        </a>
      )}
    </div>
  );
};

export default ApiKeyInput; 