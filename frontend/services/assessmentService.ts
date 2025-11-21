import { WordData, AssessmentResult } from '../types';

// 1. Load URL from environment variables
// Note: If using Create React App, use process.env.REACT_APP_ASSESSMENT_API_URL
const API_BASE_URL = import.meta.env.VITE_ASSESSMENT_API_URL || 'http://127.0.0.1:8000';

/**
 * Sends audio and target word to the backend for analysis.
 */
export const assessPronunciation = async (
  audioBlob: Blob,
  targetWord: WordData
): Promise<AssessmentResult> => {

  // 2. Create FormData to send file + text
  const formData = new FormData();

  // 'audio' matches the parameter name in your FastAPI backend: audio: UploadFile = File(...)
  // We add a filename 'recording.wav' so the backend recognizes it as a file
  formData.append('audio', audioBlob, 'recording.wav');

  // 'word' matches the parameter name in backend: word: str = Form(...)
  formData.append('word', targetWord.word);

  try {
    // 3. Send POST request
    const response = await fetch(`${API_BASE_URL}/assess`, {
      method: 'POST',
      body: formData,
      // Note: Do NOT set Content-Type header manually when using FormData.
      // The browser automatically sets it to 'multipart/form-data' with the correct boundary.
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend Error (${response.status}): ${errorText}`);
    }

    // 4. Parse response
    const data = await response.json();

    // 5. Transform backend response to frontend interface (AssessmentResult)
    // Based on the Python backend structure we built earlier:
    // {
    //   "phones": { "AH": -0.5, ... },
    //   "stress": { "truth": [0,1], "infer": [0,1] },
    //   "overall_score": 85.0
    // }

    const result: AssessmentResult = {
      word: data.word,
      overallScore: data.overall_score,
      phonemesScore: data.phones,
      stress: {
        truth: data.stress.truth,
        infer: data.stress.infer,
        syllableCount: data.stress.syllable_count
      },
      // Logic to determine pass/fail booleans based on data
      // (You can tune these thresholds)
      isPhonemesCorrect: data.overall_score > 70,
      isStressCorrect: JSON.stringify(data.stress.truth) === JSON.stringify(data.stress.infer)
    };

    return result;

  } catch (error) {
    console.error("Assessment Service Error:", error);
    throw error; // Re-throw so the UI can handle the alert
  }
};
