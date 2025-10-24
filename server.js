// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// 🔥 LOREBOOK CONFIGURATION
const ENABLE_LOREBOOK = process.env.ENABLE_LOREBOOK === 'true'; // Set to true to enable lorebook
let LOREBOOK_DATA = [];

// Load all lorebook JSON files from the lorebooks directory
function loadLorebooks() {
  if (!ENABLE_LOREBOOK) return;
  
  const lorebooksDir = path.join(__dirname, 'lorebooks');
  
  // Check if lorebooks directory exists
  if (!fs.existsSync(lorebooksDir)) {
    console.log('Lorebooks directory not found. Create a "lorebooks" folder and add JSON files.');
    return;
  }

  // Read all JSON files from the directory
  const files = fs.readdirSync(lorebooksDir).filter(file => file.endsWith('.json'));
  
  console.log(`Found ${files.length} lorebook file(s)`);
  
  files.forEach(file => {
    try {
      const filePath = path.join(lorebooksDir, file);
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      
      if (data.entries) {
        LOREBOOK_DATA.push({
          title: data.title || file,
          author: data.author || 'Unknown',
          entries: data.entries
        });
        console.log(`Loaded lorebook: ${data.title || file} (${Object.keys(data.entries).length} entries)`);
      }
    } catch (error) {
      console.error(`Error loading ${file}:`, error.message);
    }
  });
}

// Load lorebooks on startup
loadLorebooks();

// Model mapping - Use actual NVIDIA NIM model names
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo': 'xai/grok-4-0709',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.1-70b-instruct',
  'gemini-pro': 'mistralai/mistral-large-2-instruct'
};

// Function to find relevant lorebook entries based on message content
function findRelevantLoreEntries(messages) {
  if (!ENABLE_LOREBOOK || LOREBOOK_DATA.length === 0) {
    return [];
  }

  const relevantEntries = [];
  const conversationText = messages.map(m => m.content).join(' ');
  const conversationLower = conversationText.toLowerCase();
  
  // Check for lorebook activation/deactivation commands
  const activeLorebooks = new Set();
  const disabledLorebooks = new Set();
  
  // Look for <LOREBOOK:title> activation tags
  const activationMatches = conversationText.match(/<LOREBOOK:([^>]+)>/gi);
  if (activationMatches) {
    activationMatches.forEach(match => {
      const title = match.replace(/<LOREBOOK:|>/gi, '').trim().toLowerCase();
      activeLorebooks.add(title);
    });
  }
  
  // Look for <DISABLE_LOREBOOK:title> deactivation tags
  const disableMatches = conversationText.match(/<DISABLE_LOREBOOK:([^>]+)>/gi);
  if (disableMatches) {
    disableMatches.forEach(match => {
      const title = match.replace(/<DISABLE_LOREBOOK:|>/gi, '').trim().toLowerCase();
      disabledLorebooks.add(title);
    });
  }
  
  // If no specific activations, all lorebooks are active by default
  const useAllLorebooks = activeLorebooks.size === 0;

  // Iterate through all loaded lorebooks
  LOREBOOK_DATA.forEach(lorebook => {
    const lorebookTitleLower = lorebook.title.toLowerCase();
    
    // Skip if this lorebook is explicitly disabled
    if (disabledLorebooks.has(lorebookTitleLower)) {
      return;
    }
    
    // Skip if specific lorebooks are activated and this isn't one of them
    if (!useAllLorebooks && !activeLorebooks.has(lorebookTitleLower)) {
      return;
    }
    
    Object.values(lorebook.entries).forEach(entry => {
      if (!entry.keys || !Array.isArray(entry.keys)) return;
      
      // Check if any keywords match the conversation
      const hasMatch = entry.keys.some(keyword => {
        const keyLower = keyword.toLowerCase();
        if (entry.case_sensitive) {
          return conversationText.includes(keyword);
        }
        return conversationLower.includes(keyLower);
      });

      if (hasMatch && entry.content) {
        relevantEntries.push({
          content: entry.content,
          comment: entry.comment || '',
          order: entry.order || 0,
          lorebook: lorebook.title
        });
      }
    });
  });

  // Sort by order (lower numbers first)
  relevantEntries.sort((a, b) => a.order - b.order);
  
  return relevantEntries;
}

// Function to inject lorebook entries into messages
function injectLorebook(messages) {
  if (!ENABLE_LOREBOOK) return messages;

  const relevantEntries = findRelevantLoreEntries(messages);
  
  if (relevantEntries.length === 0) return messages;

  // Create lorebook context
  const lorebookContext = relevantEntries
    .map(entry => entry.content)
    .join('\n\n');

  // Find the system message or create one
  const systemMessageIndex = messages.findIndex(m => m.role === 'system');
  
  if (systemMessageIndex >= 0) {
    // Append to existing system message
    messages[systemMessageIndex].content += '\n\n[Lorebook Context]\n' + lorebookContext;
  } else {
    // Create new system message at the beginning
    messages.unshift({
      role: 'system',
      content: '[Lorebook Context]\n' + lorebookContext
    });
  }

  return messages;
}

// Health check endpoint
app.get('/health', (req, res) => {
  const totalEntries = LOREBOOK_DATA.reduce((sum, lb) => sum + Object.keys(lb.entries || {}).length, 0);
  
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    lorebook_enabled: ENABLE_LOREBOOK,
    lorebooks_loaded: LOREBOOK_DATA.length,
    total_entries: totalEntries,
    lorebook_titles: LOREBOOK_DATA.map(lb => lb.title)
  });
});

// Alternative endpoint that accepts API key as query parameter for proxy chaining
app.post('/proxy/v1/chat/completions', async (req, res) => {
  // This endpoint uses the NIM_API_KEY from environment, ignoring any auth headers
  req.url = '/v1/chat/completions';
  return app._router.handle(req, res, () => {});
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    // Accept any API key from the request - we'll use our own NIM_API_KEY
    // This allows Sophia's Lorebook to forward requests without key validation errors
    let { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Inject lorebook entries if enabled
    messages = injectLorebook(messages);
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = '';
                  }
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Transform NIM response to OpenAI format with reasoning
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Lorebook: ${ENABLE_LOREBOOK ? 'ENABLED' : 'DISABLED'}`);
  if (ENABLE_LOREBOOK) {
    const totalEntries = LOREBOOK_DATA.reduce((sum, lb) => sum + Object.keys(lb.entries || {}).length, 0);
    console.log(`Lorebooks loaded: ${LOREBOOK_DATA.length} with ${totalEntries} total entries`);
  }
});
