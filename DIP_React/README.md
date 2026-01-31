# DIP React GUI - Deployment Guide

This project is a React Single Page Application (Vite).

## 🚀 Deploying to Render

Since this folder (`DIP_React`) is inside your main repository, you must configure Render to look in the right place.

### 1. New Static Site
In Render, create a **New Static Site**.

### 2. Connect Repository
Connect your GitHub repository: `[Your Repo Name]`

### 3. Build Configuration (CRITICAL)
Fill in these settings exactly:

| Setting | Value |
| :--- | :--- |
| **Root Directory** | `DIP_React` |
| **Build Command** | `npm install && npm run build` |
| **Publish Directory** | `dist` |

> **Note**: If you leave "Root Directory" empty, the build will fail because it cannot find `package.json`.

### 4. Deploy
Click **Create Static Site**. Render will:
1.  Enter `DIP_React/`
2.  Install dependencies (`npm install`)
3.  Build the app (`vite build`)
4.  Serve the `dist/` folder.

## 🛠 Local Development
If you install Node.js locally:
```bash
cd DIP_React
npm install
npm run dev
```
