import logging
import queue
import threading
import time
import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import your agent framework components
from deal_agent_framework import DealAgentFramework
from log_utils import reformat

load_dotenv(override=True)

# --- LOGGING HANDLER ---
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def html_for(log_data):
    """Generates a scrollable terminal-style view for logs."""
    output = "<br>".join(log_data[-20:])  # Show last 20 lines
    return f"""
    <div style="height: 400px; overflow-y: auto; font-family: monospace; font-size: 12px; 
                color: #00ff00; background-color: #1a1a1a; padding: 10px; border-radius: 5px;">
    {output}
    </div>
    """

# --- MAIN APP CLASS ---
class App:
    def __init__(self):
        self.agent_framework = None

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def table_for(self, opps):
        """Converts Opportunity objects into Dataframe rows."""
        if not opps:
            return []
        return [
            [
                opp.deal.product_description,
                f"${opp.deal.price:,.2f}",
                f"${opp.estimate:,.2f}",
                f"${opp.discount:,.2f}",
                opp.deal.url,
            ]
            for opp in opps
        ]

    def get_plot(self):
        """Generates the 3D Vector Space visualization."""
        try:
            documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)
            fig = go.Figure(data=[go.Scatter3d(
                x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
                mode="markers",
                marker=dict(size=2, color=colors, opacity=0.7),
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                    aspectratio=dict(x=2.2, y=2.2, z=1),
                ),
                height=400, margin=dict(r=5, b=5, l=5, t=5),
            )
            return fig
        except Exception:
            fig = go.Figure()
            fig.update_layout(title="Vector DB Loading...")
            return fig

    def run_with_logging(self, initial_log_data):
        """A generator that runs the agent in a thread and yields logs + table updates."""
        log_queue = queue.Queue()
        result_queue = queue.Queue()
        setup_logging(log_queue)

        # Worker thread to run the actual AI logic
        def worker():
            self.get_agent_framework().run()
            result_queue.put(self.table_for(self.get_agent_framework().memory))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        active = True
        while active:
            new_logs = False
            # Drain all current logs from the queue
            try:
                while True:
                    msg = log_queue.get_nowait()
                    initial_log_data.append(reformat(msg))
                    new_logs = True
            except queue.Empty:
                pass

            # Check if processing is finished
            final_table = None
            try:
                final_table = result_queue.get_nowait()
                active = False
            except queue.Empty:
                pass

            if new_logs or not active:
                # Yield the LIVE memory so the table populates as it finds deals
                current_table = final_table or self.table_for(self.get_agent_framework().memory)
                yield initial_log_data, html_for(initial_log_data), current_table
            
            time.sleep(0.2)

    def do_select(self, selected_index: gr.SelectData):
        """Allows user to click a row to send a manual alert."""
        opportunities = self.get_agent_framework().memory
        row = selected_index.index[0]
        if row < len(opportunities):
            opportunity = opportunities[row]
            self.get_agent_framework().planner.messenger.notify(
                opportunity.deal.product_description,
                opportunity.deal.price,
                opportunity.estimate,
                opportunity.deal.url
            )

    def launch(self):
        theme = gr.themes.Soft(primary_hue="green", font=["Inter", "sans-serif"])
        
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
            log_state = gr.State([])

            gr.Markdown("<h1 style='text-align: center;'>🎯 The Price is Right</h1>")
            gr.Markdown("<p style='text-align: center;'>Autonomous Bargain Hunter Agent Coordination Framework</p>")

            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Product Description", "Price", "Estimate", "Discount", "URL"],
                    datatype=["str", "str", "str", "str", "str"],
                    wrap=True,
                    column_widths=[5, 1, 1, 1, 2],
                    interactive=False
                )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🛠️ Agent Activity Logs")
                    logs_view = gr.HTML(value=html_for(["Initializing Framework..."]))
                with gr.Column(scale=1):
                    gr.Markdown("### 🌐 Vector Space (RAG Knowledge)")
                    plot_view = gr.Plot(value=self.get_plot())

            # Start scanning on page load
            ui.load(
                self.run_with_logging,
                inputs=[log_state],
                outputs=[log_state, logs_view, opportunities_dataframe]
            )

            # Timer to run every 5 minutes (300 seconds)
            timer = gr.Timer(value=300)
            timer.tick(
                self.run_with_logging,
                inputs=[log_state],
                outputs=[log_state, logs_view, opportunities_dataframe]
            )

            opportunities_dataframe.select(self.do_select)

        # Gradio 6.0 compatibility: Theme and title go in launch()
        ui.launch(inbrowser=True, theme=theme)

if __name__ == "__main__":
    App().launch()