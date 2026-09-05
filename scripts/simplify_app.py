from pathlib import Path

content = Path('app.py').read_text(encoding='utf-8')

new_code = """def assessment_page(models, recommended_key: str) -> None:
    st.markdown("### 🍌 Assessment Studio")
    st.markdown("Upload an image or take a photo to instantly compare predictions across all four algorithms.")
    
    col1, col2 = st.columns(2)
    with col1:
        upload = st.file_uploader("Upload Banana Image", type=["jpg", "png", "jpeg"])
    with col2:
        cam = st.camera_input("Or Take a Photo")
        
    source = upload or cam
    if not source:
        st.info("Awaiting image input...")
        return
        
    try:
        image = decode_image(source.getvalue())
    except Exception as e:
        st.error(f"Invalid image: {e}")
        return
        
    st.image(image, caption="Input Image", width=300)
    st.divider()
    
    st.subheader("Algorithm Comparison (Mode A)")
    cols = st.columns(len(models))
    
    for i, (key, (pipeline, _)) in enumerate(models.items()):
        record = pipeline.predict(image, "input.jpg")
        with cols[i]:
            st.markdown(f"**{pipeline.display_name}**")
            st.metric("Prediction", record.predicted_class.title())
            st.caption(f"Confidence: {record.confidence:.1%}")
            if key == recommended_key:
                st.success("✨ Recommended Model")
"""

start = content.find('def assessment_summary(')
end = content.find('def confusion_figure(')

if start != -1 and end != -1:
    Path('app.py').write_text(content[:start] + new_code + '\n' + content[end:], encoding='utf-8')
    print('Simplified assessment_page successfully.')
else:
    print('Failed to find boundaries.')
