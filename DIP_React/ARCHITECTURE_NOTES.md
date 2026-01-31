# Architecture Status: Frontend vs Backend

## Current State
You are correct: **The React App (`DIP_React`) is currently NOT connected to a Python Backend.**

### Why?
1.  **Performance**: The "Visual Perception" module (Mach bands, brightness) was implemented directly in JavaScript/CSS. This makes interactions **instant** (0ms latency), whereas a backend connection would introduce lag.
2.  **Deployment**: As a static frontend, it can be hosted for **free** on Render's Static Site tier.
3.  **Efficiency**: For Chapter 2 (visual illusions), we don't strictly *need* Python's heavy image processing libraries yet. JavaScript can handle pixel manipulation for these demos.

## The Gap
The Python code you have in `DIP_GUI` (using `opencv`, `numpy`) is **not** being used by the React app. The React app is "mimicking" that logic.

## Options Moving Forward

### Option A: Keep it Client-Side (Frontend Only)
*   **Best for**: Visual demos, simple simulations, high interaction speed.
*   **Pros**: Free hosting, no server crashes, instant UI.
*   **Cons**: Cannot reuse your existing Python code. Logic must be rewritten in JS.

### Option B: Connect to Python (Full Stack)
*   **Best for**: Complex processing (FFTs, Histogram rendering, Morphological ops) in later chapters.
*   **Pros**: Reuses your `DIP_GUI` logic.
*   **How**:
    1.  Create `api.py` (FastAPI or Flask).
    2.  Deploy `api.py` as a **Web Service** on Render.
    3.  Update React to `fetch()` data from this API.
*   **Cons**: Slower (network lag), more complex deployment.

## Recommendation
For **Chapter 2 (Visual Perception)**, **Option A** is superior because visual illusions break if there is lag.
For future chapters (FFT, Restoration), we will likely need **Option B**.
