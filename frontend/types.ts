
export enum WorkflowStep {
  IDLE = 'IDLE',
  UPLOAD = 'UPLOAD',
  EXTRACT = 'EXTRACT',
  ANALYZE = 'ANALYZE',
  INDEX = 'INDEX',
  READY = 'READY',
  RETRIEVING = 'RETRIEVING',
  GENERATING = 'GENERATING',
  EVALUATING = 'EVALUATING'
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  judgeResult?: boolean;
  judgeFeedback?: string;
  sources?: Source[];
}

export interface Source {
  document_id: string;
  document_name: string;
  page: number | null;
  relevance_score: number | null;
  content_excerpt: string;
}

export interface EvaluationResult {
  question: string;
  answer: string;
  score: number;
  reasoning: string;
  faithfulness?: number | null;
  answer_correctness?: number | null;
  answer_relevancy?: number | null;
  context_precision?: number | null;
  context_recall?: number | null;
  ground_truth?: string;
}

export interface DocumentData {
  name: string;
  size: number;
  type: string;
  content: string;
  summary?: string;
  evaluations?: EvaluationResult[];
}

export interface WorkflowStatus {
  step: WorkflowStep;
  progress: number;
  message: string;
}
