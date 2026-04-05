from .agent import ConvAgent

def print_conv_beginning(
    agent1: ConvAgent, 
    agent2: ConvAgent, 
    max_num_turns: int
    ):
    """Print the beginning of the conversation."""
    print("\n" + "="*80)
    print("🤖 Two-Agent Conversation")
    print("="*80)
    print(f"Starting conversation between Agent {agent1.agent_role} and Agent {agent2.agent_role}")
    print(f"Total turns: {max_num_turns}")
    print("="*80)
    write_text = ("\n" + "="*80 + "\n" + 
                "🤖 Two-Agent Conversation" + "\n" + 
                "="*80 + "\n" + 
                f"Starting conversation between Agent {agent1.agent_role} and Agent {agent2.agent_role}" + "\n" + 
                f"Total turns: {max_num_turns}" + "\n" + 
                "="*80 + "\n")
    return write_text

def print_agent_message(agent_role: str, message: str, turn_number: int, message_number: int):
    """Print a formatted message from an agent."""
    if message_number == 1:
        print(f"{'='*80}")
        print(f"🔄 TURN {turn_number}")
        print(f"{'='*80}")
    print(f"{'─'*60}")
    print(f"🤖 AGENT {agent_role}")
    print(f"{message}")
    print(f"{'─'*60}")
    if message_number == 1:
        write_text = ("\n" + "="*80 + "\n" + 
                    f"🔄 TURN {turn_number}" + "\n" + 
                    "="*80 + "\n" + 
                    f"{'─'*60}" + "\n" + 
                    f"🤖 AGENT {agent_role}" + "\n" + 
                    f"{message}" + "\n" + 
                    f"{'─'*60}" + "\n")
    else:
        write_text = (f"{'─'*60}" + "\n" + 
                    f"🤖 AGENT {agent_role}" + "\n" + 
                    f"{message}" + "\n" + 
                    f"{'─'*60}" + "\n")
    return write_text

def print_conv_terminated():
    """Print the terminated conversation."""
    print("\n" + "="*80)
    print("🤖 CONVERSATION TERMINATED")
    print("="*80)
    write_text = ("\n" + "="*80 + "\n" + 
                "🤖 CONVERSATION TERMINATED" + "\n" + 
                "="*80 + "\n")
    return write_text

def print_conv_completed():
    """Print the completed conversation."""
    print("\n" + "="*80)
    print("✅ CONVERSATION COMPLETED")
    print("="*80)
    write_text = ("\n" + "="*80 + "\n" + 
                "✅ CONVERSATION COMPLETED" + "\n" + 
                "="*80 + "\n")
    return write_text