from pathlib import Path
import re

content = Path('app.py').read_text(encoding='utf-8')

new_code = """def evaluation_page(summary, models) -> None:
    st.markdown("### 📊 Model Evaluation")
    if not summary:
        st.warning("Run `scripts/train_compare.py` to generate evaluation results.")
        return

    st.subheader("Benchmark Leaderboard")
    rows = []
    for rank, item in enumerate(summary["results"], 1):
        test = item["test_metrics"]
        rows.append({
            "Rank": rank,
            "Algorithm": item["algorithm"],
            "Test Accuracy": f"{test['accuracy']:.2%}",
            "Macro-F1": f"{test['macro_f1']:.2%}"
        })
    import pandas as pd
    st.table(pd.DataFrame(rows).set_index("Rank"))

    st.subheader("Confusion Matrix Explorer")
    selected = st.selectbox("Select Pipeline", [item["algorithm"] for item in summary["results"]])
    item = next(result for result in summary["results"] if result["algorithm"] == selected)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.pyplot(confusion_figure(item["test_metrics"]["confusion_matrix"], f"{selected}"), width="stretch")
    with col2:
        report = pd.DataFrame(item["test_metrics"]["classification_report"]).T.loc[list(CLASS_NAMES), ["precision", "recall", "f1-score"]]
        st.dataframe(report.style.format("{:.3f}"), width="stretch")
"""

start = content.find('def evaluation_page(')
end = content.find('def training_page(')

if start != -1 and end != -1:
    Path('app.py').write_text(content[:start] + new_code + '\n' + content[end:], encoding='utf-8')
    print('Simplified evaluation_page successfully.')
else:
    print('Failed to find boundaries.')
