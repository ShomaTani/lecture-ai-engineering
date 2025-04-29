# ui.py
import streamlit as st
import pandas as pd
import time
from database import save_to_db, get_chat_history, get_db_count, clear_db
from llm import generate_response
from data import create_sample_evaluation_data
from metrics import get_metrics_descriptions

# --- チャットページのUI ---
def display_chat_page(pipe):
    """Display the chat page UI"""
    st.subheader("Enter your question")
    user_question = st.text_area(
        "Question",
        key="question_input",
        height=100,
        value=st.session_state.get("current_question", "")
    )
    submit_button = st.button("Send question")

    # セッション状態の初期化（安全のため）
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "current_answer" not in st.session_state:
        st.session_state.current_answer = ""
    if "response_time" not in st.session_state:
        st.session_state.response_time = 0.0
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = False

    # 質問が送信された場合
    if submit_button and user_question:
        st.session_state.current_question = user_question
        st.session_state.current_answer = ""  # reset answer
        st.session_state.feedback_given = False  # reset feedback state

        with st.spinner("The model is generating a response"):
            answer, response_time = generate_response(pipe, user_question)
            st.session_state.current_answer = answer
            st.session_state.response_time = response_time
            st.rerun()

    # Display response and feedback form if needed
    if st.session_state.current_question and st.session_state.current_answer:
        st.subheader("Response:")
        st.markdown(st.session_state.current_answer)
        st.info(f"Response time: {st.session_state.response_time:.2f}s")

        if not st.session_state.feedback_given:
            display_feedback_form()
        else:
            if st.button("Next question"):
                st.session_state.current_question = ""
                st.session_state.current_answer = ""
                st.session_state.response_time = 0.0
                st.session_state.feedback_given = False
                st.rerun()


def display_feedback_form():
    """Display the feedback input form"""
    with st.form("feedback_form"):
        st.subheader("Feedback")
        feedback_options = ["Accurate", "Partially accurate", "Inaccurate"]
        feedback = st.radio(
            "Rate the response",
            feedback_options,
            key="feedback_radio",
            label_visibility="collapsed",
            horizontal=True
        )
        correct_answer = st.text_area(
            "More accurate answer (optional)",
            key="correct_answer_input",
            height=100
        )
        feedback_comment = st.text_area(
            "Comment (optional)",
            key="feedback_comment_input",
            height=100
        )
        submitted = st.form_submit_button("Submit feedback")
        if submitted:
            is_correct = 1.0 if feedback == "Accurate" else (0.5 if feedback == "Partially accurate" else 0.0)
            combined_feedback = feedback
            if feedback_comment:
                combined_feedback += f": {feedback_comment}"

            save_to_db(
                st.session_state.current_question,
                st.session_state.current_answer,
                combined_feedback,
                correct_answer,
                is_correct,
                st.session_state.response_time
            )
            st.session_state.feedback_given = True
            st.success("Feedback has been saved!")
            st.rerun()


# --- 履歴閲覧ページのUI ---
def display_history_page():
    """Display the history and metrics page UI"""
    st.subheader("Chat History and Metrics")
    history_df = get_chat_history()

    if history_df.empty:
        st.info("No chat history available yet.")
        return

    tab1, tab2 = st.tabs(["History", "Metrics Analysis"])

    with tab1:
        display_history_list(history_df)

    with tab2:
        display_metrics_analysis(history_df)


def display_history_list(history_df):
    """Display the history list"""
    st.write("#### History List")
    filter_options = {
        "Show All": None,
        "Accurate only": 1.0,
        "Partially accurate only": 0.5,
        "Inaccurate only": 0.0
    }
    display_option = st.radio(
        "Filter display",
        options=list(filter_options.keys()),
        horizontal=True,
        label_visibility="collapsed"
    )

    filter_value = filter_options[display_option]
    if filter_value is not None:
        filtered_df = history_df[
            history_df["is_correct"].notna() &
            (history_df["is_correct"] == filter_value)
        ]
    else:
        filtered_df = history_df

    if filtered_df.empty:
        st.info("No history matches the selected criteria.")
        return

    items_per_page = 5
    total_items = len(filtered_df)
    total_pages = (total_items + items_per_page - 1) // items_per_page
    current_page = st.number_input(
        'Page',
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_df = filtered_df.iloc[start_idx:end_idx]

    for _, row in paginated_df.iterrows():
        with st.expander(f"{row['timestamp']} - Q: {row['question'][:50] if row['question'] else 'N/A'}..."):
            st.markdown(f"**Q:** {row['question']}")
            st.markdown(f"**A:** {row['answer']}")
            st.markdown(f"**Feedback:** {row['feedback']}")
            if row['correct_answer']:
                st.markdown(f"**Correct A:** {row['correct_answer']}")

            st.markdown("---")
            cols = st.columns(3)
            cols[0].metric("Accuracy score", f"{row['is_correct']:.1f}")
            cols[1].metric("Response time (s)", f"{row['response_time']:.2f}")
            cols[2].metric("Word count", f"{row['word_count']}")

            cols = st.columns(3)
            cols[0].metric("BLEU", f"{row['bleu_score']:.4f}" if pd.notna(row['bleu_score']) else "-")
            cols[1].metric("Similarity", f"{row['similarity_score']:.4f}" if pd.notna(row['similarity_score']) else "-")
            cols[2].metric("Relevance", f"{row['relevance_score']:.4f}" if pd.notna(row['relevance_score']) else "-")

    st.caption(f"Displaying items {start_idx+1} - {min(end_idx, total_items)} of {total_items}")


def display_metrics_analysis(history_df):
    """Display analysis of evaluation metrics"""
    st.write("#### Metrics Analysis")

    analysis_df = history_df.dropna(subset=['is_correct'])
    if analysis_df.empty:
        st.warning("No evaluation data available for analysis.")
        return

    accuracy_labels = {1.0: 'Accurate', 0.5: 'Partially accurate', 0.0: 'Inaccurate'}
    analysis_df['Accuracy'] = analysis_df['is_correct'].map(accuracy_labels)

    # Distribution of accuracy
    st.write("##### Distribution of accuracy")
    accuracy_counts = analysis_df['Accuracy'].value_counts()
    if not accuracy_counts.empty:
        st.bar_chart(accuracy_counts)
    else:
        st.info("No accuracy data available.")

    # Response time vs other metrics
    st.write("##### Relationship between response time and other metrics")
    metric_options = ["bleu_score", "similarity_score", "relevance_score", "word_count"]
    valid_metric_options = [
        m for m in metric_options
        if m in analysis_df.columns and analysis_df[m].notna().any()
    ]

    if valid_metric_options:
        metric_option = st.selectbox(
            "Select metric to compare",
            valid_metric_options,
            key="metric_select"
        )

        chart_data = analysis_df[['response_time', metric_option, 'Accuracy']].dropna()
        if not chart_data.empty:
            st.scatter_chart(
                chart_data,
                x='response_time',
                y=metric_option,
                color='Accuracy',
            )
        else:
            st.info(f"No valid data for selected metric ({metric_option}) and response time.")
    else:
        st.info("No metric data available for comparison with response time.")

    # Overall metrics statistics
    st.write("##### Overall metrics statistics")
    stats_cols = ['response_time', 'bleu_score', 'similarity_score', 'word_count', 'relevance_score']
    valid_stats_cols = [
        c for c in stats_cols
        if c in analysis_df.columns and analysis_df[c].notna().any()
    ]
    if valid_stats_cols:
        metrics_stats = analysis_df[valid_stats_cols].describe()
        st.dataframe(metrics_stats)
    else:
        st.info("No metrics data available to compute statistics.")

    # Average scores by accuracy level
    st.write("##### Average scores by accuracy level")
    if valid_stats_cols and 'Accuracy' in analysis_df.columns:
        try:
            accuracy_groups = analysis_df.groupby('Accuracy')[valid_stats_cols].mean()
            st.dataframe(accuracy_groups)
        except Exception as e:
            st.warning(f"Error computing averages by accuracy level: {e}")
    else:
        st.info("No data available to calculate average scores by accuracy level.")

    # Efficiency score (accuracy / (response time + 0.1))
    st.write("##### Efficiency score (accuracy / (response time + 0.1))")
    if 'response_time' in analysis_df.columns and analysis_df['response_time'].notna().any():
        analysis_df['efficiency_score'] = analysis_df['is_correct'] / (analysis_df['response_time'].fillna(0) + 0.1)
        if 'id' in analysis_df.columns:
            top_efficiency = analysis_df.sort_values('efficiency_score', ascending=False).head(10)
            if not top_efficiency.empty:
                st.bar_chart(top_efficiency.set_index('id')['efficiency_score'])
            else:
                st.info("No efficiency score data available.")
        else:
            st.bar_chart(
                analysis_df.sort_values('efficiency_score', ascending=False)
                .head(10)['efficiency_score']
            )
    else:
        st.info("No response time data available to calculate efficiency score.")


# --- サンプルデータ管理ページのUI ---
def display_data_page():
    """Display the sample data management page UI"""
    st.subheader("Sample Evaluation Data Management")
    count = get_db_count()
    st.write(f"There are currently {count} records in the database.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add sample data", key="create_samples"):
            create_sample_evaluation_data()
            st.rerun()
    with col2:
        if st.button("Clear database", key="clear_db_button"):
            if clear_db():
                st.rerun()

    st.subheader("Evaluation Metrics Descriptions")
    metrics_info = get_metrics_descriptions()
    for metric, description in metrics_info.items():
        with st.expander(f"{metric}"):
            st.write(description)
