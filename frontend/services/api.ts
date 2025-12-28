const API_URL = 'http://localhost:8000';

export interface EvaluationResult {
  question: string;
  answer: string;
  score: number;
  reasoning: string;
}

export const uploadDocuments = async (files: File[]) => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file); // 'files' must match the argument name in FastAPI
  });

  const response = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to upload document');
  }

  return response.json();
};

export const answerQuestion = async (query: string, history: Array<{ role: string; content: string }>) => {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, history }),
  });

  if (!response.ok) {
    throw new Error('Failed to get answer');
  }

  const data = await response.json();
  return data;
};

export const runEvaluations = async (context?: string) => {
  const response = await fetch(`${API_URL}/evaluate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ context: context || "" }),
  });

  if (!response.ok) {
    throw new Error('Failed to run evaluations');
  }

  return response.json();
};
