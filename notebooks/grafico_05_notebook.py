fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Grafico 1: Barras lado a lado ---
x = range(len(comparacao))
width = 0.35
bars_real = axes[0].bar([i - width/2 for i in x], comparacao["Real"], width,
                         label="Real", color="#3b82f6", alpha=0.8)
bars_pred = axes[0].bar([i + width/2 for i in x], comparacao["Previsto"], width,
                         label="Previsto pela IA", color="#f97316", alpha=0.8)

axes[0].set_xlabel("Musica (indice)")
axes[0].set_ylabel("Popularity Score")
axes[0].set_title("Real vs Previsto -- Cada Musica")
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(i+1) for i in x])
axes[0].legend()
axes[0].set_ylim(0, 105)

# --- Grafico 2: Scatter (pontos) ---
axes[1].scatter(comparacao["Real"], comparacao["Previsto"],
                c="#8b5cf6", s=100, alpha=0.7, edgecolors="white", linewidth=1.5)

# Linha de previsao perfeita (diagonal)
axes[1].plot([0, 100], [0, 100], "--", color="#22c55e", linewidth=2, label="Previsao perfeita")

# Faixa de margem (+/- 15 pontos)
axes[1].fill_between([0, 100], [0-15, 100-15], [0+15, 100+15],
                      alpha=0.1, color="#22c55e", label="Margem +/- 15 pontos")

axes[1].set_xlabel("Popularity Real")
axes[1].set_ylabel("Popularity Previsto pela IA")
axes[1].set_title("Scatter: Quao perto a IA chegou?")
axes[1].set_xlim(-5, 105)
axes[1].set_ylim(-5, 105)
axes[1].legend(loc="upper left")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.savefig(project_root / "notebooks" / "regression_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSe todos os pontos estivessem na linha verde, a IA seria perfeita!")
print("Pontos dentro da faixa verde = previsao razoavel (erro <= 15 pontos)")
     