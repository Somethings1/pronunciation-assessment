import { AssessmentResult, WordData, PhonemeError } from "../types";

// MOCK SERVICE: Simulates the behavior of the Python Backend
// 50% chance of PASS (Perfect or just Substitutions/Minor Stress diffs)
// 50% chance of FAIL (Deletions, Insertions, or Primary Stress mismatch)

export const assessPronunciation = async (
  audioBlob: Blob,
  targetWord: WordData
): Promise<AssessmentResult> => {
  
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 1000));

  const isPass = Math.random() > 0.5;
  const targetPhonemes = targetWord.phonemes.split(' ');
  let detectedStress = targetWord.stressPattern;
  const currentErrors: PhonemeError[] = [];
  const feedbackParts: string[] = [];

  // --- GENERATE MOCK DATA ---

  if (isPass) {
    // PASS SCENARIO
    // 1. Stress: Ensure Primary Stress (1) is in the same place.
    //    We might randomly change a 0 to 2 or 2 to 0 to show we ignore non-primary.
    const stressArr = detectedStress.split('');
    const secondaryIndices = stressArr.map((s, i) => s !== '1' ? i : -1).filter(i => i !== -1);
    if (secondaryIndices.length > 0 && Math.random() > 0.5) {
        const idx = secondaryIndices[0];
        stressArr[idx] = stressArr[idx] === '0' ? '2' : '0';
        detectedStress = stressArr.join('');
        feedbackParts.push("Primary stress is correct.");
    } else {
        feedbackParts.push("Stress timing is perfect.");
    }

    // 2. Phonemes: MUST BE PERFECT. No substitutions.
    feedbackParts.push("Phonemes are accurate.");

  } else {
    // FAIL SCENARIO
    // Randomly fail Stress (Primary), Phonemes (Any Error), or Both.
    const failStress = Math.random() > 0.3;
    const failPhonemes = !failStress || Math.random() > 0.5;

    if (failStress) {
        // Move the '1' to a wrong position
        const stressArr = detectedStress.split('');
        const primaryIdx = stressArr.indexOf('1');
        if (primaryIdx !== -1) {
            stressArr[primaryIdx] = '0';
            // Move to next available or 0
            const newIdx = (primaryIdx + 1) % stressArr.length;
            stressArr[newIdx] = '1';
            detectedStress = stressArr.join('');
        }
        feedbackParts.push("Primary stress was placed on the wrong syllable.");
    } else {
        feedbackParts.push("Primary stress is correct.");
    }

    if (failPhonemes) {
        const idx = Math.floor(Math.random() * targetPhonemes.length);
        const errorTypeRand = Math.random();

        if (errorTypeRand < 0.33) {
            // Deletion
            currentErrors.push({
                index: idx,
                type: 'deletion',
                expected: targetPhonemes[idx],
                actual: ''
            });
            feedbackParts.push("You missed a sound.");
        } else if (errorTypeRand < 0.66) {
            // Insertion
            currentErrors.push({
                index: idx,
                type: 'insertion',
                expected: targetPhonemes[idx],
                actual: 'AH'
            });
            feedbackParts.push("An extra sound was inserted.");
        } else {
            // Substitution
            currentErrors.push({
                index: idx,
                type: 'substitution',
                expected: targetPhonemes[idx],
                actual: 'UH' // Mock substitution
            });
            feedbackParts.push("A sound was substituted.");
        }
    } else {
         feedbackParts.push("Phonemes match reasonably well.");
    }
  }

  // --- ASSESSMENT LOGIC ---
  
  // 1. Stress Check: Compare index of '1'
  const targetPrimaryIndex = targetWord.stressPattern.indexOf('1');
  const detectedPrimaryIndex = detectedStress.indexOf('1');
  const isStressCorrect = targetPrimaryIndex === detectedPrimaryIndex;

  // 2. Phoneme Check: Fail on ANY error (Deletion, Insertion, OR Substitution).
  const isPhonemesCorrect = currentErrors.length === 0;

  return {
    isStressCorrect,
    isPhonemesCorrect,
    detectedPhonemes: targetPhonemes, // Simplified for mock display
    detectedStressPattern: detectedStress,
    phonemeErrors: currentErrors,
    feedback: feedbackParts.join(" ")
  };
};