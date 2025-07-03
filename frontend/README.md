# Frontend for Boundary Evaluation App

This is the React frontend for the Boundary Evaluation App. It allows users to upload gold and predicted boundary files, visualize evaluation metrics, and interact with various parts of the backend API.

## Prerequisites

- **Node.js** (v18.x or later recommended)
- **npm** (v9.x or later) or **yarn**

## Getting Started

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**

   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure Backend URL**

   By default, the frontend expects the backend API at `/api`.  
   If your backend is running elsewhere (e.g., `http://127.0.0.1:8000/api`), create a `.env` file in the frontend directory and set:

   ```
   REACT_APP_BACKEND=http://127.0.0.1:8000/api
   ```

4. **Start the development server**

   ```bash
   npm start
   # or
   yarn start
   ```

   This will run the app in development mode at [http://localhost:3000](http://localhost:3000).

   - The frontend will hot-reload on changes.
   - Make sure your backend server is also running and accessible.

5. **Build for production**

   To create a production build:

   ```bash
   npm run build
   # or
   yarn build
   ```

   The build artifacts will be output to the `build/` directory.

## Usage

- Upload your **gold** and **output** boundary files (in JSON format) in the evaluation component.
- View metrics such as **Precision, Recall, F1**, and error analyses.
- Other components let you visualize graphs, train embeddings, or explore other backend features.

## Notes

- This frontend is designed to work with the corresponding FastAPI backend (see `backend/` folder).
- API endpoints and file upload requirements are described in the backend README.

## Troubleshooting

- **CORS errors:** Make sure your backend allows requests from this frontend's origin.
- **API errors:** Check your backend logs and ensure it is running and accessible at the configured URL.
- **File upload issues:** Ensure your file format matches backend requirements (`.json` for boundary evaluation).
