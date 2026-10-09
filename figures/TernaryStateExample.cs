// TernaryStateExample.cs  (Unity only)
// Minimal MonoBehaviour showing how a game would hold and react to a TernaryState.
// Attach to any GameObject; the state is editable in the Inspector and survives in
// save data as four numbers (A, B, C, Actions).

#if UNITY_5_3_OR_NEWER
using UnityEngine;
using TernaryIndexing;

public class TernaryStateExample : MonoBehaviour
{
    public TernaryState state = new TernaryState();

    void OnEnable()  { state.CampChanged += OnCampChanged; }
    void OnDisable() { state.CampChanged -= OnCampChanged; }

    // Call these from gameplay code (dialogue choices, quest outcomes, ...).
    public void ChoseA()       { state.Toward(Camp.A); Report(); }
    public void ChoseB()       { state.Toward(Camp.B); Report(); }
    public void ChoseC()       { state.Toward(Camp.C); Report(); }
    public void ChoseAB()      { state.Blend(Camp.A, Camp.B); Report(); }
    public void ChoseBalance() { state.Balance(); Report(); }
    public void MajorChoiceA() { state.Toward(Camp.A, weight: 3f); Report(); }

    void OnCampChanged(Camp from, Camp to)
    {
        // Hook transition effects here (one-off event when the camp flips).
        Debug.Log($"Camp changed {from} -> {to}");
    }

    void Report()
    {
        // Content lookups use the derived values, never a stored copy of them.
        Debug.Log(state.ToString());
    }
}
#endif
