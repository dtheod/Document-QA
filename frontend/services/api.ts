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

// The evaluation takes many minutes, so the backend runs it as a background job:
// start it, then poll until it is done
const EVALUATION_POLL_MS = 5000;

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

  let job = await response.json();
  while (job.status === 'running') {
    await new Promise((resolve) => setTimeout(resolve, EVALUATION_POLL_MS));
    const status = await fetch(`${API_URL}/evaluate`);
    if (!status.ok) {
      throw new Error('Failed to get evaluation status');
    }
    job = await status.json();
  }

  if (job.status !== 'done') {
    throw new Error(job.error || 'Evaluation failed');
  }
  return job.results;
};
