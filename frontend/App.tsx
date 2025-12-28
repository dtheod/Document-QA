import React, { useState, useRef, useEffect } from 'react';
import { WorkflowStep } from './types';
import { Message, DocumentData } from '../types';
import WorkflowGraph from './components/WorkflowGraph';
import EvaluationTab from './components/EvaluationTab';
import { uploadDocuments, answerQuestion, runEvaluations } from './services/api';

type Tab = 'chat' | 'evaluations';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('chat');
  const [workflow, setWorkflow] = useState<WorkflowStep>(WorkflowStep.IDLE);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [docs, setDocs] = useState<DocumentData[]>([]);
  const [loading, setLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, activeTab]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setWorkflow(WorkflowStep.UPLOAD);
    setLoading(true);

    try {
      const fileList = Array.from(files);

      // Simulation of Upload/Extract
      setWorkflow(WorkflowStep.EXTRACT);
      // Wait a bit for UI transition
      await new Promise(r => setTimeout(r, 500));

      setWorkflow(WorkflowStep.ANALYZE);
      const response = await uploadDocuments(fileList);

      // Indexing
      setWorkflow(WorkflowStep.INDEX);

      const newDocs: DocumentData[] = fileList.map(file => ({
        name: file.name,
        size: file.size,
        type: file.type,
        content: "Processed on server",
        summary: "Indexed successfully",
        evaluations: undefined
      }));

      setDocs(prev => [...prev, ...newDocs]);

      setWorkflow(WorkflowStep.READY);
      setMessages(prev => [{
        id: Date.now().toString(),
        role: 'assistant',
        content: `Hello! I've successfully indexed ${fileList.length} documents:\n\n${fileList.map(f => `- **${f.name}**`).join('\n')}\n\nYou can now ask me any questions based on these documents.`,
        timestamp: new Date()
      }]);
    } catch (err) {
      console.error(err);
      setWorkflow(WorkflowStep.IDLE);
    } finally {
      setLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    if (docs.length === 0) return;

    setWorkflow(WorkflowStep.EVALUATING);
    setLoading(true);

    try {
      // Evaluation runs on the "current context" which implies all docs or we iterate?
      // Without specific doc selection, we might run generic evaluation or need backend support for "session".
      // For now, let's just trigger it and attach to the first doc (or dummy) for UI purposes or run against "All"

      // The current backend endpoint takes 'context', but implementation might rely on indexed vector store.
      // Let's pass empty context to imply "use vector store" logic if backend supports it.

      const evaluations = await runEvaluations("");
      // Attaching to the first doc for display purposes if we don't have a "global" evaluation tab
      // Or we can just display them.

      // For simplicity in this iteration: attach to the most recent doc or just store globally?
      // Let's update the last added doc with these evaluations for now.
      setDocs(prev => {
        if (prev.length === 0) return prev;
        const newDocs = [...prev];
        newDocs[newDocs.length - 1].evaluations = evaluations;
        return newDocs;
      });

    } catch (err) {
      console.error("Evaluation failed", err);
    } finally {
      setWorkflow(WorkflowStep.READY);
      setLoading(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || docs.length === 0 || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setWorkflow(WorkflowStep.RETRIEVING);

    try {
      setWorkflow(WorkflowStep.GENERATING);
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const response = await answerQuestion(input, history);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer || "I'm sorry, I couldn't generate an answer.",
        timestamp: new Date(),
        judgeResult: response.judge_result,
        judgeFeedback: response.judge_feedback
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
    } finally {
      setWorkflow(WorkflowStep.READY);
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden font-sans text-slate-900">
      {/* Sidebar */}
      <aside className={`bg-white border-r border-slate-200 transition-all duration-300 flex flex-col ${isSidebarOpen ? 'w-80' : 'w-0 overflow-hidden'}`}>
        <div className="p-6 border-b border-slate-100">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <span className="bg-blue-600 text-white p-1 rounded">DM</span>
            DocuMind AI
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          <section>
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Active Workflow</h3>
            <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-3">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${workflow === WorkflowStep.READY ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' : 'bg-blue-500 animate-pulse'}`}></div>
                <span className="text-sm font-medium">{workflow.replace('_', ' ')}</span>
              </div>
              <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-blue-600 h-full transition-all duration-500"
                  style={{ width: `${(Object.values(WorkflowStep).indexOf(workflow) + 1) * 11.1}%` }}
                />
              </div>
            </div>
          </section>

          {docs.length > 0 && (
            <section>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Documents ({docs.length})</h3>
                <button onClick={() => setDocs([])} className="text-xs text-red-500 hover:text-red-700">Clear</button>
              </div>
              <div className="space-y-3">
                {docs.map((d, i) => (
                  <div key={i} className="p-3 bg-blue-50 rounded-xl border border-blue-100">
                    <div className="flex items-center gap-3">
                      <div className="bg-blue-100 p-2 rounded-lg shrink-0">
                        <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                      </div>
                      <div className="overflow-hidden min-w-0">
                        <p className="text-sm font-semibold truncate" title={d.name}>{d.name}</p>
                        <p className="text-xs text-slate-500">{(d.size / 1024).toFixed(1)} KB</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          <div className="pt-4">
            <label className="w-full py-2.5 px-4 rounded-xl border border-slate-200 text-sm font-medium hover:bg-slate-50 transition-colors flex items-center justify-center gap-2 text-slate-600 cursor-pointer">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Documents
              <input type="file" className="hidden" accept=".txt,.pdf,.md" multiple onChange={handleFileUpload} />
            </label>

          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative min-w-0">
        {/* Toggle Sidebar Button */}
        <button
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          className="absolute left-4 top-4 z-20 p-2 bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50 transition-colors"
        >
          <svg className={`w-5 h-5 transition-transform ${isSidebarOpen ? '' : 'rotate-180'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>

        {/* Header / Workflow Viz */}
        <header className="bg-white border-b border-slate-200 pt-16 pb-4">
          <WorkflowGraph currentStep={workflow} />

          {docs.length > 0 && (
            <div className="flex justify-center mt-2 px-4">
              <div className="flex bg-slate-100 p-1 rounded-xl">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`px-6 py-1.5 rounded-lg text-sm font-semibold transition-all ${activeTab === 'chat' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                >
                  Chat
                </button>
                <button
                  onClick={() => setActiveTab('evaluations')}
                  className={`px-6 py-1.5 rounded-lg text-sm font-semibold transition-all ${activeTab === 'evaluations' ? 'bg-white text-blue-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
                >
                  Evaluations
                </button>
              </div>
            </div>
          )}
        </header>

        {/* Tab Content Area */}
        <div className="flex-1 flex flex-col min-h-0 bg-slate-50">
          {docs.length === 0 && workflow === WorkflowStep.IDLE ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
              <div className="w-20 h-20 bg-blue-100 text-blue-600 rounded-3xl flex items-center justify-center mb-6 shadow-xl shadow-blue-100">
                <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold mb-2">Upload your documents</h2>
              <p className="text-slate-500 max-w-md mb-8">
                Drop your PDF or TXT files here. We'll analyze, index, and prepare them for instant Q&A and automated evaluations.
              </p>

              <label className="cursor-pointer group">
                <div className="px-8 py-4 bg-blue-600 text-white rounded-2xl font-semibold shadow-lg shadow-blue-200 group-hover:bg-blue-700 group-hover:scale-105 transition-all flex items-center gap-3">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Select Documents
                </div>
                <input type="file" className="hidden" accept=".txt,.pdf,.md" multiple onChange={handleFileUpload} />
              </label>
            </div>
          ) : (
            <>
              {activeTab === 'chat' ? (
                <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
                  <div
                    ref={scrollRef}
                    className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth"
                  >
                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${message.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-white border border-slate-200'
                          }`}>
                          <div className="text-sm whitespace-pre-wrap leading-relaxed">
                            {message.content}
                          </div>
                          {message.judgeResult !== undefined && (
                            <div className={`mt-3 pt-3 border-t ${message.role === 'user' ? 'border-blue-400/30' : 'border-slate-100'}`}>
                              <div className="flex items-center gap-2 mb-1">
                                {message.judgeResult ? (
                                  <span className="flex items-center gap-1 text-xs font-semibold text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-100">
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                                    Verified Answer
                                  </span>
                                ) : (
                                  <span className="flex items-center gap-1 text-xs font-semibold text-amber-600 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-100">
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                                    Needs Verification
                                  </span>
                                )}
                              </div>
                              {message.judgeFeedback && (
                                <p className={`text-xs ${message.role === 'user' ? 'text-blue-100' : 'text-slate-500'} italic`}>
                                  "{message.judgeFeedback}"
                                </p>
                              )}
                            </div>
                          )}
                          <div className={`text-[10px] mt-2 font-medium opacity-60 ${message.role === 'user' ? 'text-blue-100 text-right' : 'text-slate-400'}`}>
                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </div>
                        </div>
                      </div>
                    ))}
                    {loading && workflow !== WorkflowStep.EVALUATING && (
                      <div className="flex justify-start">
                        <div className="bg-white border border-slate-200 rounded-2xl p-4 flex gap-2">
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                          <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Input Area */}
                  <div className="p-6 bg-white border-t border-slate-200">
                    <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto flex gap-4">
                      <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={docs.length > 0 ? "Ask a question about your documents..." : "Indexing documents..."}
                        disabled={docs.length === 0 || loading}
                        className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all disabled:opacity-50"
                      />
                      <button
                        type="submit"
                        disabled={docs.length === 0 || loading || !input.trim()}
                        className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 transition-all shadow-md shadow-blue-200 disabled:opacity-50 disabled:shadow-none"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                      </button>
                    </form>
                  </div>
                </div>
              ) : (
                <EvaluationTab
                  evaluations={docs.length > 0 ? (docs[docs.length - 1].evaluations || []) : []}
                  isLoading={workflow === WorkflowStep.EVALUATING}
                  onRunEvaluation={handleRunEvaluation}
                />
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
