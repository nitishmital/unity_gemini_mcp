import tkinter as tk
from tkinter import ttk, scrolledtext
import asyncio
from mcp_core import MCPCore
import threading
import queue

class MCPGui:
    def __init__(self, server_script: str):
        self.root = tk.Tk()
        self.root.title("Gemini MCP Chat")
        self.root.geometry("800x600")
        
        self.core = MCPCore()
        self.server_script = server_script
        self.message_queue = queue.Queue()
        
        self._setup_ui()
        self._setup_async()
        
    def _setup_ui(self):
        # Chat display
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=20)
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Input area
        input_frame = ttk.Frame(self.root)
        input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)
        
        self.input_field = ttk.Entry(input_frame)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        send_button = ttk.Button(input_frame, text="Send", command=self._handle_send)
        send_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        self.input_field.bind("<Return>", lambda e: self._handle_send())
        
    def _setup_async(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()
        
    def _run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())
        self.loop.run_forever()
        
    async def _connect(self):
        connected = await self.core.connect(self.server_script)
        if not connected:
            self._append_message("System", "Failed to connect to MCP server")
            return
        self._append_message("System", "Connected to MCP server")
        
    def _handle_send(self):
        query = self.input_field.get().strip()
        if not query:
            return
            
        self.input_field.delete(0, tk.END)
        self._append_message("You", query)
        
        asyncio.run_coroutine_threadsafe(
            self._process_query(query),
            self.loop
        )
        
    async def _process_query(self, query: str):
        response = await self.core.process_request(query)
        self.message_queue.put(("Assistant", response))
        self.root.after(0, self._check_message_queue)
        
    def _check_message_queue(self):
        while not self.message_queue.empty():
            sender, message = self.message_queue.get()
            self._append_message(sender, message)
            
    def _append_message(self, sender: str, message: str):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, f"\n{sender}:\n{message}\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.see(tk.END)
        
    def run(self):
        self.root.mainloop()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        asyncio.run(self.core.cleanup())
