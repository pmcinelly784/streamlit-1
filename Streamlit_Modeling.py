import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simulate_markov_chain(P, initial_distribution, periods):
    distributions = [initial_distribution]
    current_distribution = initial_distribution

    for _ in range(periods):
        current_distribution = np.dot(current_distribution, P)
        distributions.append(current_distribution)

    return np.array(distributions)

def main():
    st.title('Simulation of Homelessness in the US')
    st.subheader("By Parker McInelly")
    st.write('This Markov Chain Simulation Dashboard is an interactive tool designed to dynamically simulate and visualize the behavior Homelessness via a Markov Chain. This dashboard allows you to input parameters and transition probabilities, and then observe how the Markov chain evolves over time.')
    url = "https://docs.google.com/presentation/d/1mps2uXCifeL6R8xem_yRNHJjRxzFTHv52z1RVzLfgp8/edit?usp=sharing"
    st.write("Check out this [link](%s) to my presentation!" % url)
    with st.sidebar:
        st.title("Change Parameters Here")
        include_subsidy = st.checkbox('Include Government Subsidized Housing', value=False)
        exclude_employed = st.checkbox('Exclude "Employed with Home"', value=True)
        periods = st.slider('Number of Periods to Simulate', min_value=1, max_value=120, value=10, step=1)
        population = st.slider('Population to Simulate', min_value=10000, max_value=1000000, value=10000, step=10000)


    st.write('We are hoping to see what the long-term effects of government-offered housing on homelessness in the US')

    states_without_subsidy = ['Employed with Home', 'At Risk of Homelessness', 'Transitionary Homelessness', 'Permanently Homeless']
    initial_distribution_without_subsidy = np.array([0.945, 0.045, 0.005, 0.005]) * population # Adjust these values based on your data

    P_without_subsidy = np.array([
    [0.9625, 0.0365, 0.0005, 0.0005],    # From Employed
    [0.9, 0.05, 0.025, 0.025],         # From At Risk
    [0.05, 0.2, 0.66, 0.09],        # From Homeless - Transition
    [0.0, 0.0, 0.2, 0.8]            # From Permanently Homeless
    ])


    states_with_subsidy = ['Employed with Home', 'At Risk of Homelessness', 'Transitionary Homelessness', 'Permanently Homeless', 'Government Subsidized Housing']
    initial_distribution_with_subsidy = np.array([0.945, 0.045, 0.005, 0.005, 0.0]) * population# Adjust these values based on your data

    P_with_subsidy = np.array([
        [0.9625, 0.0365, 0.0005, 0.000, 0.0005],    # From Employed
        [0.90, 0.05, 0.02, 0.02, 0.01],          # From At Risk
        [0.05, 0.2, 0.61, 0.09, 0.05],         # From Homeless - Transition
        [0.0, 0.0, 0.05, 0.9, 0.05],            # From Permanently Homeless
        [0.1, 0.1, 0.2, 0.1, 0.5]             # From Government Subsidized Housing (stay in subsidized housing)
    ])




    if include_subsidy:
        P = P_with_subsidy
        states = states_with_subsidy
        initial_distribution = initial_distribution_with_subsidy
    else:
        P = P_without_subsidy
        states = states_without_subsidy
        initial_distribution = initial_distribution_without_subsidy
    
    st.subheader('This is the Probability Matrix being simulated:')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("This is a transposed Markov Matrix, each row beign a different economic/housing state.")
        st.data_editor(P)
    row_sums = np.sum(P, axis=1)
    with col2:
        st.write("This is the sum of the Rows. Each row should equal 1 to maintain a constant population.")
        st.table(row_sums)

    # Simulate me Markov chain
    distributions = simulate_markov_chain(P, initial_distribution, periods)


    # Population Plot
    population_dist = distributions
    plt.figure(figsize=(10, 6))
    for i, state in enumerate(states):
        if not (exclude_employed and state == 'Employed with Home'):
            plt.plot(population_dist[:, i], label=state)

    plt.title(f'Markov Chain Simulation With a population of {population}')
    plt.xlabel('Periods')
    plt.ylabel(f'Number of People')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    col1, col2 = st.columns(2)

    with col1:
        first_dist = population_dist[0]
        st.subheader('\nInitial Population:')
        for state, proportion in zip(states, first_dist):
            if not (exclude_employed and state == 'Employed with Home'):
                st.write(f'{state}: {round(proportion)}')

    with col2:
        final_distribution = population_dist[-1]
        st.subheader('\n\nFinal Population:')
        for state, proportion in zip(states, final_distribution):
            if not (exclude_employed and state == 'Employed with Home'):
                st.write(f'{state}: {round(proportion)}')





    # data = {
    # 'Type': ['Without Government Subsidies', 'With Government Subsidies'],
    # 'Average Tax Burden (%)': [14.9, 16.0]
    # }

    # df = pd.DataFrame(data)
    # fig, ax = plt.subplots()
    # bars = ax.bar(df['Type'], df['Average Tax Burden (%)'], color=['#1f77b4', '#ff7f0e'])

    # for bar in bars:
    #     height = bar.get_height()
    #     ax.annotate(f'{height:.1f}%',
    #                 xy=(bar.get_x() + bar.get_width() / 2, height),
    #                 xytext=(0, 3), 
    #                 textcoords="offset points",
    #                 ha='center', va='bottom')

    # ax.set_xlabel('Type')
    # ax.set_ylabel('Average Tax Burden (%)')
    # ax.set_title('Average Tax Burden Comparison')

    # # Display the plot in Streamlit
    # st.title('Average Tax Burden Comparison')
    # st.pyplot(fig)
    # monthly_cost_per_person = population / (round(final_distribution[2] + final_distribution[3], 0)*1500)
    # saved_people = round(final_distribution[2] + final_distribution[3], 0)

    # st.write(
    #     f"For a 1.1% Tax increase on the population of {population:,}, "
    #     f"or about $500 annually per person, "
    #     f"we can save {saved_people:,.0f} homeless people and give them what they need to get back on their feet again."
    # )



if __name__ == '__main__':
    main()
