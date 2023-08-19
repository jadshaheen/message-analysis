import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import sys

from collections import defaultdict


def get_messages(conn):
    messages_query = """SELECT *, 
                        datetime(date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch', 'localtime')  as date_utc 
                        FROM message"""
    messages = pd.read_sql_query(messages_query, conn)
    messages["message_date"] = messages["date"]
    messages["timestamp"] = messages["date_utc"].apply(lambda x: pd.Timestamp(x))
    messages["date"] = messages["timestamp"].apply(lambda x: x.date())
    messages["month"] = messages["timestamp"].apply(lambda x: int(x.month))
    messages["year"] = messages["timestamp"].apply(lambda x: int(x.year))
    messages["minute"] = messages["timestamp"].apply(lambda x: int(x.minute))
    messages["hour"] = messages["timestamp"].apply(lambda x: int(x.hour))
    messages["year_month"] = messages["timestamp"].dt.to_period("M")
    messages.rename(columns={"ROWID": "message_id"}, inplace=True)
    return messages


def get_handles(conn):
    handles = pd.read_sql_query("SELECT * FROM handle", conn)
    handles.rename(columns={"id": "sender", "ROWID": "handle_id"}, inplace=True)
    return handles


def get_chat_message_joins(conn):
    chat_message_joins = pd.read_sql_query("SELECT * FROM chat_message_join", conn)
    return chat_message_joins


def merge_data(messages, handles, chat_message_joins):
    merge_level_1 = pd.merge(
        messages[
            [
                "text",
                "handle_id",
                "date",
                "message_date",
                "timestamp",
                "month",
                "year",
                "year_month",
                "hour",
                "minute",
                "is_sent",
                "message_id",
            ]
        ],
        handles[["handle_id", "sender"]],
        on="handle_id",
        how="left",
    )

    df_messages = pd.merge(
        merge_level_1,
        chat_message_joins[["chat_id", "message_id"]],
        on="message_id",
        how="left",
    )

    return df_messages


def remove_no_contact_senders(df_messages):
    """
    Remove rows from a DataFrame where the sender is not a contact,
    i.e. the value is a phone number or email address.
    """
    mask = ~df_messages["sender"].str.contains("@|^\+|^\d")
    return df_messages[mask]


def get_dm_chat_ids(df_messages):
    """
    Isolate the chat id that corresponds to each contact's DM conversation.
    Returns a map of contact to DM chat id.
    """
    ids_to_senders = dict()
    banned_ids = set()
    for index, row in df_messages.iterrows():
        chat_id = row["chat_id"]
        sender = row["sender"]
        if not ids_to_senders.get(chat_id):
            ids_to_senders[chat_id] = sender
        elif ids_to_senders[chat_id] != sender:
            banned_ids.add(chat_id)
    for id in banned_ids:
        del ids_to_senders[id]

    senders_to_ids = {sender: id for id, sender in ids_to_senders.items()}
    return senders_to_ids


def get_group_chat_ids(df_messages):
    """
    Similar to `get_dm_chat_ids` but returns the set of id's that represent
    gropu chats, to be used to filter the dataframe down to only DM's.
    """

    ids_to_senders = dict()
    banned_ids = set()
    for index, row in df_messages.iterrows():
        chat_id = row["chat_id"]
        sender = row["sender"]
        if not ids_to_senders.get(chat_id):
            ids_to_senders[chat_id] = sender
        elif ids_to_senders[chat_id] != sender:
            banned_ids.add(chat_id)

    return banned_ids


def filter_groups(df_messages):
    """
    Get rid of all group chat messages, so we can see data just based on DM's
    """

    banned_ids = get_group_chat_ids(df_messages)
    filtered_df = df_messages[~df_messages["chat_id"].isin(banned_ids)]
    return filtered_df


def filter_funusha(df_messages):
    """
    Don't consider Funusha messages. This is to get more accurate values for my top senders
    that aren't corrupted by Sirjan constantly spamming Funusha.
    """
    mask = ~(df_messages["chat_id"] == 12)
    return df_messages[mask]


def single_chat(df_messages, chat_id):
    """
    Filter to only messages from a specific group chat.
    Requires the chat_id of the group
    """
    return df_messages[df_messages["chat_id"] == chat_id]


def get_messages_per_year(df_messages):
    """Groups messages by year and returns a list of strings indicating the number of messages for each year.

    Example:
        >>> messages = pd.DataFrame({'text': ['hi', 'hello', 'how are you'], 'sender': ['John', 'Jane', 'John'], 'timestamp': ['2022-01-01', '2022-02-01', '2023-01-01']})
        >>> get_messages_per_year(messages)
        ['Messages in 2022: 2', 'Messages in 2023: 1']
    """
    result = []
    messages_per_year = df_messages.groupby("year").size()
    for year, count in messages_per_year.items():
        result.append("Messages in {}: {}".format(year, count))
    return result


def get_messages_per_number(df_messages):
    return df_messages["sender"].value_counts().sort_values(ascending=False)


def plot_messages_by_sender_over_time(df_messages, sender):
    # get the top 20 message senders
    # top_senders = get_messages_per_number(df_messages).head(1)

    # filter df_messages to only top senders
    # df_top_senders_messages = df_messages[df_messages["sender"].isin(top_senders.index)]

    df_one_sender = df_messages[df_messages["sender"] == sender]

    df_sender_counts = df_one_sender.groupby("year_month")["message_id"].count()

    # USE
    # df_sender_month = df_messages.groupby(['sender', 'year_month'])['message_id'].count()

    fig, ax = plt.subplots()
    df_sender_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Number of Messages")
    ax.set_title("Messages Sent by " + sender)
    plt.show()


def get_top_senders_by_month(df_messages, top_x):
    # group by year_month and sender and aggregate on message counts
    grouped = df_messages.groupby(["year_month", "sender"]).agg({"message_id": "count"})

    # sort data
    sorted_data = grouped.sort_values(
        ["year_month", "message_id"], ascending=[True, False]
    )

    return sorted_data.groupby("year_month").head(top_x)


def compute_sender_scores(top_senders_by_month):
    senders_to_scores = defaultdict(lambda: 0)
    cur_year_month = ""
    cur_rank = 0
    for year_month, sender in top_senders_by_month.index:
        if year_month != cur_year_month:
            cur_rank = 0
            cur_year_month = year_month
        senders_to_scores[sender] += 5 - cur_rank
        cur_rank += 1

    sender_scores = [
        key + ": " + str(value)
        for key, value in sorted(
            senders_to_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]
    return sender_scores


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: Please provide the path to your chat.db file.")
    sys.exit(1)

    message_database = sys.argv[1]
    conn = sqlite3.connect(message_database)

    messages = get_messages(conn)
    handles = get_handles(conn)
    chat_message_joins = get_chat_message_joins(conn)

    df_messages = merge_data(messages, handles, chat_message_joins)
    df_messages = remove_no_contact_senders(df_messages)

    # uncomment the following line to exclude group message data
    # df_messages = filter_groups(df_messages)

    print("\nTOP 20 SENDERS\n")

    print(get_messages_per_number(df_messages).head(20))

    print("\nTOP 5 SENDERS PER MONTH")
    print(get_top_senders_by_month(df_messages, 5).to_string())
