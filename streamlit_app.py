
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
# Load the data


uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type='csv')
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    team_list = data['teamname'].unique().tolist()

# Streamlit 앱 시작
st.title('Team Champion Analysis')



def heatma(data):
    features = ['wardsplaced', 'wardskilled', 'gamelength', 'result', 
                'firstdragon', 'firstbaron', 'kills', 'assists', 'teamkills',
                'golddiffat10', 'golddiffat15', 'xpdiffat10', 'xpdiffat15',
                'towers', 'firstmidtower', 'firsttothreetowers',
                'deaths', 'teamdeaths', 'firsttower']

    # 상관관계 계산
    correlation_matrix = data[features].corr()

    # 히트맵 그리기
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".3f")
    plt.title("Feature Correlation Matrix")
    st.pyplot()



def ward_patterns_percentage(data, teamname):
    team_data = data[data['teamname'] == teamname]
    
    # 해당 팀의 평균 와드 설치/제거 수
    avg_wards_placed_team = team_data['wardsplaced'].mean()
    avg_wards_killed_team = team_data['wardskilled'].mean()
    
    # 전체 데이터의 평균 와드 설치/제거 수
    avg_wards_placed_all = data['wardsplaced'].mean()
    avg_wards_killed_all = data['wardskilled'].mean()
    
    # 전체 데이터의 팀별 평균 와드 설치/제거 수
    all_teams_avg_wards_placed = data.groupby('teamname')['wardsplaced'].mean().values
    all_teams_avg_wards_killed = data.groupby('teamname')['wardskilled'].mean().values
    
    # 백분위 수 계산
    wards_placed_percentile = 100 - (np.searchsorted(np.sort(all_teams_avg_wards_placed), avg_wards_placed_team) / len(all_teams_avg_wards_placed) * 100)
    wards_killed_percentile = 100 - (np.searchsorted(np.sort(all_teams_avg_wards_killed), avg_wards_killed_team) / len(all_teams_avg_wards_killed) * 100)
    
    return avg_wards_placed_team, avg_wards_placed_all, wards_placed_percentile, avg_wards_killed_team, avg_wards_killed_all, wards_killed_percentile

#print(ward_patterns_percentage(data,'Klanik Esport'))



def visualize_ward_patterns(team_data, ward_results):
    avg_wards_placed_team, avg_wards_placed_all, wards_placed_percentile, avg_wards_killed_team, avg_wards_killed_all, wards_killed_percentile = ward_results
    
    # 시각화
    categories = ['Wards Placed', 'Wards Killed']
    team_values = [avg_wards_placed_team, avg_wards_killed_team]
    all_values = [avg_wards_placed_all, avg_wards_killed_all]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, team_values, width, label=team_data['teamname'].iloc[0], color=['blue', 'green'])
    rects2 = ax.bar(x + width/2, all_values, width, label='All Teams Avg', color=['lightblue', 'lightgreen'])
    
    # 백분위 수 추가
    ax.text(0 - width/2, avg_wards_placed_team + 0.5, f"{wards_placed_percentile:.1f}%", ha='center')
    ax.text(1 - width/2, avg_wards_killed_team + 0.5, f"{wards_killed_percentile:.1f}%", ha='center')
    
    ax.set_ylabel('Average Count')
    ax.set_title('Average Wards Placed and Killed by Teams')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    fig.tight_layout()
    st.pyplot()

# 사용 예:
#team_data = data[data['teamname'] == 'Klanik Esport']
#ward_results = ward_patterns_percentage(data, 'Klanik Esport')
#visualize_ward_patterns(team_data, ward_results)



def game_length_win_rate_percentage(data, teamname):
    team_data = data[data['teamname'] == teamname]
    
    # 전체 데이터를 기준으로 중간 게임 길이 계산
    median_length = data['gamelength'].median()
    
    short_games = team_data[team_data['gamelength'] < median_length]
    long_games = team_data[team_data['gamelength'] >= median_length]
    
    short_win_rate_team = short_games['result'].mean()
    long_win_rate_team = long_games['result'].mean()
    
    # 전체 데이터의 짧은 게임 및 긴 게임 승률 계산
    short_win_rate_all = data[data['gamelength'] < median_length]['result'].mean()
    long_win_rate_all = data[data['gamelength'] >= median_length]['result'].mean()
    
    # 전체 데이터의 팀별 짧은 게임 및 긴 게임 승률 계산
    all_teams_short_win_rate = data[data['gamelength'] < median_length].groupby('teamname')['result'].mean().values
    all_teams_long_win_rate = data[data['gamelength'] >= median_length].groupby('teamname')['result'].mean().values
    
    # 백분위 수 계산
    short_win_rate_percentile = 100 - (np.searchsorted(np.sort(all_teams_short_win_rate), short_win_rate_team) / len(all_teams_short_win_rate) * 100)
    long_win_rate_percentile = 100 - (np.searchsorted(np.sort(all_teams_long_win_rate), long_win_rate_team) / len(all_teams_long_win_rate) * 100)
    
    return short_win_rate_team, short_win_rate_all, short_win_rate_percentile, long_win_rate_team, long_win_rate_all, long_win_rate_percentile

#print(game_length_win_rate_percentage(data,'Klanik Esport'))


def visualize_game_length_win_rate(team_data, game_length_results):
    short_win_rate_team, short_win_rate_all, short_win_rate_percentile, long_win_rate_team, long_win_rate_all, long_win_rate_percentile = game_length_results
    
    # 시각화
    categories = ['Short Games Win Rate', 'Long Games Win Rate']
    team_values = [short_win_rate_team * 100, long_win_rate_team * 100]  # Percentage 변환
    all_values = [short_win_rate_all * 100, long_win_rate_all * 100]  # Percentage 변환
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, team_values, width, label=team_data['teamname'].iloc[0], color=['blue', 'green'])
    rects2 = ax.bar(x + width/2, all_values, width, label='All Teams Avg', color=['lightblue', 'lightgreen'])
    
    # 백분위 수 추가
    ax.text(0 - width/2, short_win_rate_team * 100 + 2, f"{short_win_rate_percentile:.1f}%", ha='center')
    ax.text(1 - width/2, long_win_rate_team * 100 + 2, f"{long_win_rate_percentile:.1f}%", ha='center')
    
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Short and Long Games Win Rate by Teams')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    fig.tight_layout()
    st.pyplot()

# 사용 예:
#team_data = data[data['teamname'] == 'Klanik Esport']
#game_length_results = game_length_win_rate_percentage(data, 'Klanik Esport')
#visualize_game_length_win_rate(team_data, game_length_results)




def first_objectives_percentage(data, teamname):
    team_data = data[data['teamname'] == teamname]
    
    first_dragon_team = team_data['firstdragon'].sum()
    first_baron_team = team_data['firstbaron'].sum()
    
    # 전체 데이터의 팀별 첫 용 및 첫 바론 획득 수 계산
    all_teams_first_dragon = data.groupby('teamname')['firstdragon'].sum().values
    all_teams_first_baron = data.groupby('teamname')['firstbaron'].sum().values
    
    # 백분위 수 계산
    first_dragon_percentile = 100 - (np.searchsorted(np.sort(all_teams_first_dragon), first_dragon_team) / len(all_teams_first_dragon) * 100)
    first_baron_percentile = 100 - (np.searchsorted(np.sort(all_teams_first_baron), first_baron_team) / len(all_teams_first_baron) * 100)
    
    return first_dragon_team, first_dragon_percentile, first_baron_team, first_baron_percentile


#print(first_objectives_percentage(data,'MS Company'))



def visualize_first_objectives(team_data, objectives_results):
    first_dragon_team, first_dragon_percentile, first_baron_team, first_baron_percentile = objectives_results
    
    # 시각화
    categories = ['First Dragon', 'First Baron']
    team_values = [first_dragon_team, first_baron_team]
    
    x = np.arange(len(categories))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x, team_values, width, label=team_data['teamname'].iloc[0], color=['blue', 'green'])
    
    # 백분위 수 추가
    ax.text(0, first_dragon_team + 0.5, f"{first_dragon_percentile:.1f}%", ha='center')
    ax.text(1, first_baron_team + 0.5, f"{first_baron_percentile:.1f}%", ha='center')
    
    ax.set_ylabel('Count')
    ax.set_title('First Dragon and Baron Obtained by Teams')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    fig.tight_layout()
    st.pyplot()

# 사용 예:
#team_data = data[data['teamname'] == 'MS Company']
#objectives_results = first_objectives_percentage(data, 'MS Company')
#visualize_first_objectives(team_data, objectives_results)



def first_objectives_win_rate_percentage(data, teamname):
    team_data = data[data['teamname'] == teamname]
    
    # 첫 오브젝트 획득 시의 승률 계산
    first_dragon_win_rate_team = team_data[team_data['firstdragon'] == 1]['result'].mean()
    first_baron_win_rate_team = team_data[team_data['firstbaron'] == 1]['result'].mean()
    first_tower_win_rate_team = team_data[team_data['firsttower'] == 1]['result'].mean()
    
    # 전체 데이터의 팀별 첫 오브젝트 획득 시의 승률 계산
    all_teams_first_dragon_win_rate = data[data['firstdragon'] == 1].groupby('teamname')['result'].mean().values
    all_teams_first_baron_win_rate = data[data['firstbaron'] == 1].groupby('teamname')['result'].mean().values
    all_teams_first_tower_win_rate = data[data['firsttower'] == 1].groupby('teamname')['result'].mean().values
    
    # 백분위 수 계산
    first_dragon_percentile = 100 - (np.searchsorted(np.sort(all_teams_first_dragon_win_rate), first_dragon_win_rate_team) / len(all_teams_first_dragon_win_rate) * 100)
    first_baron_percentile = 100 - (np.searchsorted(np.sort(all_teams_first_baron_win_rate), first_baron_win_rate_team) / len(all_teams_first_baron_win_rate) * 100)
    first_tower_percentile = 100 - (np.searchsorted(np.sort(all_teams_first_tower_win_rate), first_tower_win_rate_team) / len(all_teams_first_tower_win_rate) * 100)
    
    return {
        'first_dragon_win_rate': first_dragon_win_rate_team,
        'first_dragon_percentile': first_dragon_percentile,
        'first_baron_win_rate': first_baron_win_rate_team,
        'first_baron_percentile': first_baron_percentile,
        'first_tower_win_rate': first_tower_win_rate_team,
        'first_tower_percentile': first_tower_percentile
    }

#print(first_objectives_win_rate_percentage(data,'MS Company'))



def visualize_first_objectives_win_rate(objectives_win_rate_results, teamname):
    categories = ['First Dragon Win Rate', 'First Baron Win Rate', 'First Tower Win Rate']
    team_values = [
        objectives_win_rate_results['first_dragon_win_rate'],
        objectives_win_rate_results['first_baron_win_rate'],
        objectives_win_rate_results['first_tower_win_rate']
    ]
    
    percentiles = [
        objectives_win_rate_results['first_dragon_percentile'],
        objectives_win_rate_results['first_baron_percentile'],
        objectives_win_rate_results['first_tower_percentile']
    ]
    
    x = np.arange(len(categories))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects = ax.bar(x, team_values, width, label=teamname, color=['blue', 'green', 'red'])
    
    # 백분위 수 추가
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate(f"{percentiles[i]:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate by First Objective Obtained')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.legend()

    fig.tight_layout()
    st.pyplot()




def kill_participation_percentage(data, teamname):
    team_data = data[data['teamname'] == teamname]
    
    # 팀별 킬 참여율 계산
    team_data['kill_participation'] = (team_data['kills'] + team_data['assists']) / team_data['teamkills']
    
    # 전체 선수들의 킬 참여율 계산
    all_players_kill_participation = data['kill_participation'] = (data['kills'] + data['assists']) / data['teamkills']
    
    # 주어진 팀의 선수별 킬 참여율 백분위 계산
    result = {}
    for player in team_data['playername'].unique():
        player_kill_participation = team_data[team_data['playername'] == player]['kill_participation'].mean()
        percentile = 100 - (np.searchsorted(np.sort(all_players_kill_participation), player_kill_participation) / len(all_players_kill_participation) * 100)
        result[player] = {
            'kill_participation': player_kill_participation,
            'percentile': percentile
        }
    
    return result

#print(kill_participation_percentage(data,'ViV Esport'))



def visualize_kill_participation(kill_participation_results, teamname):
    players = list(kill_participation_results.keys())
    kill_participation_values = [kill_participation_results[player]['kill_participation'] for player in players]
    percentiles = [kill_participation_results[player]['percentile'] for player in players]

    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x=players, y=kill_participation_values, palette='viridis')

    # 백분위 수 정보 추가
    for i, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f"{percentiles[i]:.1f}%", ha="center", va="bottom")

    plt.ylim(0, 1.1)
    plt.ylabel('Kill Participation Rate')
    plt.xlabel('Player Name')
    plt.title(f'Player Kill Participation Rate for {teamname}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

# 사용 예:
#kill_participation_results = kill_participation_percentage(data, 'ViV Esport')
#visualize_kill_participation(kill_participation_results, 'ViV Esport')


def team_gold_and_xp_read_percentage(data, teamname):
    team_data = data[data['teamname'] == teamname]
    
    # 주어진 팀의 10분과 15분 시점의 평균 골드와 경험치 차이 계산
    team_avg_gold_diff_at_10 = team_data['golddiffat10'].mean()
    team_avg_gold_diff_at_15 = team_data['golddiffat15'].mean()
    team_avg_xp_diff_at_10 = team_data['xpdiffat10'].mean()
    team_avg_xp_diff_at_15 = team_data['xpdiffat15'].mean()
    
    # 백분위 계산
    gold_diff_at_10_percentile = 100 - (np.searchsorted(np.sort(data['golddiffat10']), team_avg_gold_diff_at_10) / len(data) * 100)
    gold_diff_at_15_percentile = 100 - (np.searchsorted(np.sort(data['golddiffat15']), team_avg_gold_diff_at_15) / len(data) * 100)
    xp_diff_at_10_percentile = 100 - (np.searchsorted(np.sort(data['xpdiffat10']), team_avg_xp_diff_at_10) / len(data) * 100)
    xp_diff_at_15_percentile = 100 - (np.searchsorted(np.sort(data['xpdiffat15']), team_avg_xp_diff_at_15) / len(data) * 100)
    
    return {
        'avg_gold_diff_at_10': team_avg_gold_diff_at_10,
        'gold_diff_at_10_percentile': gold_diff_at_10_percentile,
        'avg_gold_diff_at_15': team_avg_gold_diff_at_15,
        'gold_diff_at_15_percentile': gold_diff_at_15_percentile,
        'avg_xp_diff_at_10': team_avg_xp_diff_at_10,
        'xp_diff_at_10_percentile': xp_diff_at_10_percentile,
        'avg_xp_diff_at_15': team_avg_xp_diff_at_15,
        'xp_diff_at_15_percentile': xp_diff_at_15_percentile
    }




#print(team_gold_and_xp_read_percentage(data,'beGenius ESC'))





def visualize_team_gold_and_xp_lead(lead_results, teamname):
    labels = ['Gold Diff at 10', 'Gold Diff at 15', 'XP Diff at 10', 'XP Diff at 15']
    values = [
        lead_results['avg_gold_diff_at_10'],
        lead_results['avg_gold_diff_at_15'],
        lead_results['avg_xp_diff_at_10'],
        lead_results['avg_xp_diff_at_15']
    ]
    percentiles = [
        lead_results['gold_diff_at_10_percentile'],
        lead_results['gold_diff_at_15_percentile'],
        lead_results['xp_diff_at_10_percentile'],
        lead_results['xp_diff_at_15_percentile']
    ]
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=labels, y=values, palette='viridis')

    # 백분위 수 정보 추가
    for i, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f"{percentiles[i]:.1f}%", ha="center", va="bottom")

    plt.ylabel('Lead Value')
    plt.xlabel('Metrics')
    plt.title(f'Gold and XP Lead Analysis for {teamname}')
    plt.tight_layout()
    st.pyplot()

# 사용 예:
#lead_results = team_gold_and_xp_read_percentage(data, 'beGenius ESC')
#visualize_team_gold_and_xp_lead(lead_results, 'beGenius ESC')


def tower_preference_percentage(data, teamname):
    team_data = data[data['teamname'] == teamname]
    
    # 해당 팀의 평균 타워 파괴 통계 계산
    avg_total_towers = team_data['towers'].mean()
    avg_first_midtowers = team_data['firstmidtower'].mean()
    avg_first_threetowers = team_data['firsttothreetowers'].mean()
    
    # 백분위 계산
    total_towers_percentile = 100 - (np.searchsorted(np.sort(data['towers']), avg_total_towers) / len(data) * 100)
    first_midtowers_percentile = 100 - (np.searchsorted(np.sort(data['firstmidtower']), avg_first_midtowers) / len(data) * 100)
    first_threetowers_percentile = 100 - (np.searchsorted(np.sort(data['firsttothreetowers']), avg_first_threetowers) / len(data) * 100)
    
    return {
        'avg_total_towers': avg_total_towers,
        'total_towers_percentile': total_towers_percentile,
        'avg_first_midtowers': avg_first_midtowers,
        'first_midtowers_percentile': first_midtowers_percentile,
        'avg_first_threetowers': avg_first_threetowers,
        'first_threetowers_percentile': first_threetowers_percentile
    }

#print(tower_preference_percentage(data, 'ViV Esport'))



def visualize_tower_preference(tower_results, teamname):
    labels = ['Total Towers', 'First Mid Towers', 'First Three Towers']
    values = [
        tower_results['avg_total_towers'],
        tower_results['avg_first_midtowers'],
        tower_results['avg_first_threetowers']
    ]
    percentiles = [
        tower_results['total_towers_percentile'],
        tower_results['first_midtowers_percentile'],
        tower_results['first_threetowers_percentile']
    ]
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=labels, y=values, palette='magma')

    # 백분위 수 정보 추가
    for i, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f"{percentiles[i]:.1f}%", ha="center", va="bottom")

    plt.ylabel('Average Tower Count')
    plt.xlabel('Tower Categories')
    plt.title(f'Tower Preference Analysis for {teamname}')
    plt.tight_layout()
    st.pyplot()

# 사용 예:
#tower_results = tower_preference_percentage(data, 'ViV Esport')
#visualize_tower_preference(tower_results, 'ViV Esport')


def team_kda_percentage(data, teamname):
    # 특정 팀의 데이터 필터링
    team_data = data[data['teamname'] == teamname]
    
    # 10분과 15분 지점에서의 킬, 데스, 어시스트 합산
    total_kills_at_10 = team_data['killsat10'].sum()
    total_deaths_at_10 = team_data['deathsat10'].sum()
    total_assists_at_10 = team_data['assistsat10'].sum()
    
    total_kills_at_15 = team_data['killsat15'].sum()
    total_deaths_at_15 = team_data['deathsat15'].sum()
    total_assists_at_15 = team_data['assistsat15'].sum()
    
    # 데스가 0인 경우 1로 변환
    if total_deaths_at_10 == 0:
        total_deaths_at_10 = 1
    
    if total_deaths_at_15 == 0:
        total_deaths_at_15 = 1
    
    # KDA 계산 
    kda_at_10 = (total_kills_at_10 + total_assists_at_10) / total_deaths_at_10
    kda_at_15 = (total_kills_at_15 + total_assists_at_15) / total_deaths_at_15
    
    # 전체 데이터를 기준으로 10분 및 15분 KDA 계산
    data['KDA_at_10'] = (data['killsat10'] + data['assistsat10']) / data['deathsat10'].replace(0, 1)
    data['KDA_at_15'] = (data['killsat15'] + data['assistsat15']) / data['deathsat15'].replace(0, 1)
    
    # 백분위 계산
    kda_at_10_percentile = 100 - (np.searchsorted(np.sort(data['KDA_at_10']), kda_at_10) / len(data) * 100)
    kda_at_15_percentile = 100 - (np.searchsorted(np.sort(data['KDA_at_15']), kda_at_15) / len(data) * 100)
    
    return {
        'kda_at_10': kda_at_10,
        'kda_at_10_percentile': kda_at_10_percentile,
        'kda_at_15': kda_at_15,
        'kda_at_15_percentile': kda_at_15_percentile
    }

#print(team_kda_percentage(data, 'beGenius ESC'))



def visualize_team_kda(kda_results, teamname):
    # 데이터 정리
    labels = ['KDA at 10', 'KDA at 15']
    values = [kda_results['kda_at_10'], kda_results['kda_at_15']]
    percentiles = [kda_results['kda_at_10_percentile'], kda_results['kda_at_15_percentile']]
    
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x=labels, y=values, palette='viridis')

    # 백분위 수 정보 추가
    for i, rect in enumerate(ax.patches):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.01, f"{percentiles[i]:.1f}%", ha="center", va="bottom")

    plt.ylabel('KDA Value')
    plt.xlabel('Time Period')
    plt.title(f'KDA Analysis for {teamname} at different time intervals')
    plt.tight_layout()
    st.pyplot()

# 사용 예:
#kda_results = team_kda_percentage(data, 'beGenius ESC')
#visualize_team_kda(kda_results, 'beGenius ESC')


#세이브 데이터
def team_champion_analysis(data, teamname):
    # 승리한 게임 필터링
    won_games = data[(data['teamname'] == teamname) & (data['result'] == 1)]
    
    # 가장 많이 사용한 챔피언 Top 5 찾기
    top_champions = won_games['champion'].value_counts().head(10).index #10개로 바꿔 봤음
    
    # Top 5 챔피언의 이름과 포지션 가져오기
    champion_positions = won_games[won_games['champion'].isin(top_champions)][['champion', 'position']].drop_duplicates()
    
    # Top 5 챔피언의 KDA 계산
    champion_kill=won_games[won_games['champion'].isin(top_champions)]['kills'].sum()
    champion_assists=won_games[won_games['champion'].isin(top_champions)]['assists'].sum()
    champion_deaths=won_games[won_games['champion'].isin(top_champions)]['deaths'].sum()
    if champion_deaths==0:
        champion_deaths=1
    else:
        champion_kda = (champion_kill+ champion_assists) / (champion_deaths)
    
    
        # 각 게임에서 Ban 당한 Top 5 챔피언들 찾기
    banned_champions_per_game = won_games[['ban1', 'ban2', 'ban3', 'ban4', 'ban5']].apply(lambda x: set(x) & set(top_champions), axis=1)
    
    # 각 게임에서 Ban 당한 Top 5 챔피언의 수 계산
    banned_counts = banned_champions_per_game.apply(len)
    
    # KDA 계산
    won_games['KDA'] = (won_games['teamkills'] + won_games['teamdeaths']) / won_games['teamdeaths'].replace(0, 1)
    
    # 벤 당한 챔피언 수에 따른 승률 및 벤 당한 챔피언들 반환
    result = {}
    for i in [1, 2, 3, 4, 5]:
        mask = (banned_counts == i)
        avg_kda = won_games.loc[mask, 'KDA'].mean()
        bans = banned_champions_per_game[mask].tolist()
        
        # 승률이 하위 3위였던 게임의 챔피언과 KDA 찾기
        lowest_winrate_games = won_games[mask].nsmallest(3, 'KDA')
        lowest_winrate_champions = lowest_winrate_games[['champion', 'KDA']].to_dict(orient='records')
        
        result[f"{i} bans"] = {
            "avg_kda": avg_kda, 
            "banned_champions": bans, 
            "low_winrate_champions": lowest_winrate_champions
        }
        #골드 획득 시점, CS 차이, 특정 오브젝트 획득 시점 등을 추가로 제공하기 위해서 gameid를 제공하자!
    return champion_positions, champion_kda, result

#print(team_champion_analysis(data,'beGenius ESC'))




def visualize_team_champion_analysis(champion_data, teamname):
    champion_positions, champion_kda, result = champion_data

    # Top champions and their positions
   # st.write("aaa")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=champion_positions['champion'], y=champion_positions.index, hue=champion_positions['position'], dodge=False)
    plt.title(f'Top Champions and Their Positions for {teamname}')
    plt.ylabel('Champion')
    plt.xlabel('Number of Selections')
    st.pyplot()

    # Top champion's average KDA
    plt.figure(figsize=(10, 5))
    sns.barplot(x=['Top Champion'], y=[champion_kda])
    plt.title(f"Top Champion's Average KDA for {teamname}")
    plt.ylabel('KDA')
    st.pyplot()

    # Banned champions count vs. average KDA
    bans = list(result.keys())
    avg_kda = [result[b]['avg_kda'] for b in bans]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=bans, y=avg_kda, palette='viridis')
    plt.title(f'Banned Champions Count vs. Average KDA for {teamname}')
    plt.ylabel('Average KDA')
    st.pyplot()
    #st.write("bbb")

# Usage:
#champion_data = team_champion_analysis(data, 'beGenius ESC')
#visualize_team_champion_analysis(champion_data, 'beGenius ESC')


def analyze_and_visualize(data, teamname):
    # 1. 팀 데이터 필터링
    
    team_data = data[data['teamname'] == teamname]
    
    # 2. 9가지 분석 실행
    
    ward_results = ward_patterns_percentage(data,teamname)
    game_length_win_rate_results = game_length_win_rate_percentage(data, teamname)
    first_objective_results = first_objectives_percentage(data, teamname)
    first_objective_win_rate_results = first_objectives_win_rate_percentage(data, teamname)
    kill_participation_results = kill_participation_percentage(data, teamname)
    team_gold_and_xp_results = team_gold_and_xp_read_percentage(data, teamname)
    tower_preference_results = tower_preference_percentage(data, teamname)
    team_kda_results = team_kda_percentage(data, teamname)
    team_champion_results = team_champion_analysis(data, teamname)
    
    # 3. 시각화
   # st.write("ward")
    visualize_ward_patterns(team_data,ward_results)
   # st.write("len")
    visualize_game_length_win_rate(team_data,game_length_win_rate_results)
   # st.write("obj")
    visualize_first_objectives(team_data,first_objective_results)
   # st.write("obj_rate")
    visualize_first_objectives_win_rate(first_objective_win_rate_results,teamname)
    #st.write("kill part")
    visualize_kill_participation(kill_participation_results,teamname)
   # st.write("gold,xp")
    visualize_team_gold_and_xp_lead(team_gold_and_xp_results,teamname)
   # st.write("tower")
    visualize_tower_preference(tower_preference_results,teamname)
   # st.write("team kda")
    visualize_team_kda(team_kda_results,teamname)
    #st.write("champ analysis")
    visualize_team_champion_analysis(team_champion_results,teamname)

    return


# Streamlit app starts here
st.title('Team Analysis')

#teamname = st.text_input('Enter the team name:', 'beGenius ESC')
#champion_positions, champion_kda, result = team_champion_analysis(data, teamname)
# 사용자에게 팀 선택하도록 함
selected_team = st.selectbox('Select a team:', team_list)

# 선택한 팀에 대한 정보 표시 (이 부분은 원하는대로 추가/수정할 수 있습니다.)
st.write(f"You selected {selected_team}")
st.write(selected_team)
if st.button('Analyze'):
    analyze_and_visualize(data, selected_team)

