
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
}

export interface EvaluationResult {
  question: string;
  answer: string;
  score: number;
  reasoning: string;
  faithfulness?: number;
  answer_correctness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_recall?: number;
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
