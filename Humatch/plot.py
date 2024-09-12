import matplotlib.pyplot as plt
import seaborn as sns


def plot_example_boxplot(df, is_human_col="is_human",
                         score_cols=["top_heavy_score", "top_light_score", "top_paired_score"],
                         fontsize=12):
    '''
    Compare the top CNN scores for human and non-human sequences with a boxplot
    
    :param df: DataFrame with columns is_human, top_heavy_score, top_light_score, top_paired_score
    :param is_human_col: str, column name with 1/0 for human/non-human
    :param score_cols: list of str, column names with float CNN scores
    :param fontsize: int, fontsize for plot
    '''
    # Melt the DataFrame
    df_melted = df.melt(id_vars=is_human_col, value_vars=score_cols,
                        var_name="CNN", value_name="Score")

    # Create the boxplot
    plt.figure(figsize=(4,2), dpi=150)
    sns.boxplot(x="CNN", y="Score", hue=is_human_col, data=df_melted, palette={1: "green", 0: "red"})

    # make labels pretty
    plt.title("Compare CNN scores", fontsize=fontsize)
    plt.xticks(range(3), ["Heavy", "Light", "Paired"], fontsize=fontsize)
    plt.yticks([0,1], fontsize=fontsize)
    plt.xlabel("")
    plt.ylabel("")

    # make legend pretty and move outside plot
    plt.legend(title="Human", loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)

    plt.show()
