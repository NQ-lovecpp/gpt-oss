#!/usr/bin/env python3
"""Simple text-based bar chart for GDP values.

This script prints the GDP of the United States, China, and Japan for the year
2023 (values are approximate, in trillion USD).  It then renders a bar chart
where each bar is proportional to the GDP value.
"""

def main():
    # GDP values for 2023 in trillion US dollars (approximate).
    gdp = {
        "United States": 25.3,
        "China": 18.5,
        "Japan": 5.0,
    }

    # Determine the maximum value to scale the bars.
    max_value = max(gdp.values())
    # Use a maximum bar length of 50 characters for readability.
    max_bar_length = 50
    scale = max_value / max_bar_length

    print("GDP Bar Chart (2023, trillion USD)")
    print("-----------------------------------")
    for country, value in gdp.items():
        bar_length = int(value / scale)
        bar = "#" * bar_length
        print(f"{country:15}: {value:6.1f} |{bar}")


if __name__ == "__main__":
    main()
