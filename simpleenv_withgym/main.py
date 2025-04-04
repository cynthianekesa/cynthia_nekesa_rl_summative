from training.dqn_training import train_dqn
from training.pg_training import train_pg

if __name__ == "__main__":
    print("Choose training method:")
    print("1: DQN")
    print("2: PPO")
    choice = input("Enter choice (1/2): ")

    if choice == "1":
        train_dqn()
    elif choice == "2":
        train_pg()
    else:
        print("Invalid choice.")
