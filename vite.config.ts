import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { spawn, ChildProcess } from 'node:child_process';
import http from 'node:http';

let pythonProcess: ChildProcess | null = null;

function pythonDaemonPlugin() {
  return {
    name: 'python-daemon',
    configureServer() {
      // Check if daemon is already running
      const req = http.get('http://127.0.0.1:5005/api/status', (res) => {
        // already running
      });

      req.on('error', () => {
        console.log('[Vite] Starting Python NumPy-GPT Daemon on port 5005...');
        pythonProcess = spawn('python3', ['daemon.py'], {
          stdio: 'inherit',
          detached: false,
        });

        process.on('exit', () => {
          if (pythonProcess) pythonProcess.kill();
        });
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    pythonDaemonPlugin(),
  ],
  server: {
    port: 3000,
    host: '0.0.0.0',
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5005',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  preview: {
    port: 3000,
    host: '0.0.0.0',
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5005',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
