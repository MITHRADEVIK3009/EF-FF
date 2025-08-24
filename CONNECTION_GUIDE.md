# Dashboard Connection Guide

## Current Issue
The Streamlit dashboard is running but cannot connect to the streaming server.

## Solution Steps

### 1. **Check Server Status**
First, verify the streaming server is running:
```bash
# Check if port 8765 is listening
netstat -an | findstr :8765
```

### 2. **Start Streaming Server (if not running)**
```bash
# Terminal 1: Start the streaming server
python streaming_server.py
```

You should see:
```
INFO:__main__:Air Quality Streaming Server running at ws://localhost:8765
INFO:__main__:Model initialized and ready for streaming data
```

### 3. **Start Dashboard**
```bash
# Terminal 2: Start the dashboard
streamlit run streamlit_dashboard.py
```

### 4. **Connect in Dashboard**
1. Open the dashboard in your browser
2. Click the "Connect" button in the sidebar
3. You should see "Connected to streaming server!"

## Troubleshooting

### **If Connection Fails:**

1. **Check Server Logs**
   - Look at the streaming server terminal for errors
   - Check if OpenAQ API is accessible

2. **Verify Port Availability**
   ```bash
   # Check if port 8765 is free
   netstat -an | findstr :8765
   ```

3. **Test WebSocket Connection**
   ```bash
   # Test with a simple WebSocket client
   python -c "
   import websockets
   import asyncio
   
   async def test():
       try:
           ws = await websockets.connect('ws://localhost:8765')
           print('WebSocket connection successful')
           await ws.close()
       except Exception as e:
           print(f'WebSocket connection failed: {e}')
   
   asyncio.run(test())
   "
   ```

### **Common Issues:**

1. **Port Already in Use**
   - Kill any existing processes on port 8765
   - Restart the streaming server

2. **Firewall Blocking**
   - Check Windows Firewall settings
   - Allow Python/Streamlit through firewall

3. **Server Not Responding**
   - Restart the streaming server
   - Check for Python errors in server terminal

## Quick Fix Commands

```bash
# Kill any existing processes on port 8765
netstat -ano | findstr :8765
taskkill /PID <PID> /F

# Restart streaming server
python streaming_server.py

# In new terminal, start dashboard
streamlit run streamlit_dashboard.py
```

## Expected Behavior

### **When Connected:**
- Status shows "Connected"
- Real-time metrics appear in sidebar
- Data starts flowing in charts
- No error messages

### **When Disconnected:**
- Status shows "Disconnected"
- Warning message: "Please connect to streaming server"
- No data displayed

## Testing Connection

1. **Start both services**
2. **Click Connect in dashboard**
3. **Check server logs for connection**
4. **Verify data is flowing**

If you still have issues, check the server logs for specific error messages.
