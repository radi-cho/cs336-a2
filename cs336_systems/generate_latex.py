import pandas as pd
import itertools

df = pd.read_csv("triton.csv")

impls = df["impl"].unique()
dtypes = df["dtype"].unique()
seq_lens = sorted(df["seq_len"].unique())
embed_dims = sorted(df["embed_dim"].unique())

def make_latex_table(sub_df, impl, dtype):
    table = []
    table.append(f"\\begin{{table}}[h!]")
    table.append(f"\\centering")
    table.append(f"\\scriptsize")
    table.append(f"\\caption{{Results for {impl}, dtype={dtype}}}")
    table.append("\\begin{tabular}{l" + "c" * len(embed_dims) + "}")
    header = ["len vs dim"] + [str(ed) for ed in embed_dims]
    table.append(" & ".join(header) + " \\\\ \\hline")

    for sl in seq_lens:
        row = [str(sl)]
        for ed in embed_dims:
            match = sub_df[(sub_df["seq_len"] == sl) & (sub_df["embed_dim"] == ed)]
            if not match.empty:
                fwd = match["fwd_ms"].values[0]
                bwd = match["bwd_ms"].values[0]
                row.append(f"({fwd:.1f}, {bwd:.1f})")
            else:
                row.append("--")
        table.append(" & ".join(row) + " \\\\")

    table.append("\\end{tabular}")
    table.append("\\end{table}")
    return "\n".join(table)

if __name__ == "__main__":
    for impl, dtype in itertools.product(impls, dtypes):
        sub_df = df[(df["impl"] == impl) & (df["dtype"] == dtype)]
        if not sub_df.empty:
            latex = make_latex_table(sub_df, impl, dtype)
            print(latex)
