def load_data(df, output_path: str):
    df.to_csv(output_path, index=False)
    print(f'Dados salvos em: {output_path}')
