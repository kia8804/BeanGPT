import React, { useState, useEffect } from 'react';
import { FaKey, FaEye, FaEyeSlash, FaCheck, FaExclamationTriangle, FaCog, FaShieldAlt, FaLock, FaPlus, FaTimes } from 'react-icons/fa';

const ApiKeyInput = ({ darkMode, onApiKeyChange }) => {
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isValidFormat, setIsValidFormat] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [keyStatus, setKeyStatus] = useState('none'); // 'none', 'valid', 'invalid_format', 'invalid_key'

  // Load API key from localStorage on component mount
  useEffect(() => {
    const savedApiKey = localStorage.getItem('openai_api_key');
    if (savedApiKey) {
      setApiKey(savedApiKey);
      const validFormat = validateApiKeyFormat(savedApiKey);
      setIsValidFormat(validFormat);
      if (validFormat) {
        setIsSaved(true);
        setKeyStatus('valid');
        onApiKeyChange(savedApiKey);
      } else {
        setKeyStatus('invalid_format');
      }
    } else {
      setKeyStatus('none');
    }
  }, [onApiKeyChange]);

  // Validate OpenAI API key format
  const validateApiKeyFormat = (key) => {
    if (!key) return false;
    // Support both old (sk-) and new (sk-proj-) formats
    const hasValidPrefix = key.startsWith('sk-') || key.startsWith('sk-proj-');
    const hasValidLength = key.length >= 50; // More realistic minimum length
    console.log('🔑 API Key validation:', { prefix: hasValidPrefix, length: key.length, valid: hasValidLength });
    return hasValidPrefix && hasValidLength;
  };

  // Handle API key input changes
  const handleApiKeyChange = (e) => {
    const value = e.target.value; // Remove .trim() to avoid key corruption
    setApiKey(value);
    const validFormat = validateApiKeyFormat(value);
    setIsValidFormat(validFormat);
    
    if (validFormat) {
      localStorage.setItem('openai_api_key', value);
      setIsSaved(true);
      setKeyStatus('valid');
      onApiKeyChange(value);
    } else if (value.length > 0) {
      setIsSaved(false);
      setKeyStatus('invalid_format');
      onApiKeyChange('');
    } else {
      setIsSaved(false);
      setKeyStatus('none');
      onApiKeyChange('');
    }
  };

  // Handle API key validation errors from backend
  const handleApiKeyError = () => {
    setKeyStatus('invalid_key');
    setIsSaved(false);
    // Don't clear the key, just mark it as invalid
    onApiKeyChange('');
  };

  // Expose error handler to parent
  useEffect(() => {
    // Store the error handler reference so parent can call it
    window.handleApiKeyError = handleApiKeyError;
    return () => {
      delete window.handleApiKeyError;
    };
  }, []);

  // Clear API key
  const clearApiKey = () => {
    setApiKey('');
    setIsValidFormat(false);
    setIsSaved(false);
    setIsExpanded(false);
    setKeyStatus('none');
    localStorage.removeItem('openai_api_key');
    onApiKeyChange('');
  };

  // Toggle expanded view
  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  // Get status display info
  const getStatusInfo = () => {
    switch (keyStatus) {
      case 'valid':
        return {
          text: 'API Key Ready',
          icon: FaShieldAlt,
          color: 'text-green-600 dark:text-green-400',
          bgColor: 'bg-gradient-to-r from-green-500 to-emerald-600',
          dotColor: 'bg-green-400'
        };
      case 'invalid_format':
        return {
          text: 'Invalid Format',
          icon: FaExclamationTriangle,
          color: 'text-red-600 dark:text-red-400',
          bgColor: 'bg-gradient-to-r from-red-500 to-red-600',
          dotColor: 'bg-red-400'
        };
      case 'invalid_key':
        return {
          text: 'Invalid API Key',
          icon: FaTimes,
          color: 'text-red-600 dark:text-red-400',
          bgColor: 'bg-gradient-to-r from-red-500 to-red-600',
          dotColor: 'bg-red-400'
        };
      default:
        return null;
    }
  };

  const statusInfo = getStatusInfo();

  return (
    <div className="flex items-center space-x-4">
      {/* Compact Status/Input Toggle */}
      {statusInfo && !isExpanded ? (
        // Saved state - compact indicator
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className={`w-6 h-6 rounded-full ${statusInfo.bgColor} flex items-center justify-center`}>
              <statusInfo.icon className="text-white text-xs" />
            </div>
            <span className={`text-sm font-medium ${statusInfo.color}`}>
              {statusInfo.text}
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
                        ? isValidFormat
                          ? keyStatus === 'valid'
                            ? 'border-green-500 focus:ring-green-500/30'
                            : 'border-yellow-500 focus:ring-yellow-500/30'
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
                        {keyStatus === 'valid' ? (
                          <FaCheck className="text-green-500 text-xs" />
                        ) : keyStatus === 'invalid_key' ? (
                          <FaTimes className="text-red-500 text-xs" />
                        ) : isValidFormat ? (
                          <FaCheck className="text-yellow-500 text-xs" />
                        ) : (
                          <FaExclamationTriangle className="text-red-500 text-xs" />
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
              
              {(isSaved || keyStatus !== 'none') && (
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
      {keyStatus === 'none' && !isExpanded && (
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse"></div>
          <span className="text-xs text-red-600 dark:text-red-400 font-medium">
            API Key Required
          </span>
        </div>
      )}
      
      {/* Get API Key Link - only show when expanded and no key */}
      {isExpanded && keyStatus === 'none' && (
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