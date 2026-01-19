# Lecture 02 - Part 1: Visualization Fundamentals

## Overview
This part covers the foundational principles of data visualization, including visual encoding, perception principles, color theory, and design best practices for creating effective visualizations.

## Key Topics

### 1. Importance and Goals of Data Visualization
- **Why Visualize Data?**
  - Humans process visual information faster than text
  - Patterns, trends, and outliers become immediately apparent
  - Supports data-driven decision making
- **Goals of Visualization**:
  - **Explore**: Discover patterns and relationships
  - **Explain**: Communicate findings effectively
  - **Confirm**: Validate hypotheses and models
- **Anscombe's Quartet**: Same statistics, different patterns - demonstrates why visualization matters

### 2. Visual Encoding Principles
- **Visual Channels** (ordered by perceptual accuracy):
  1. Position (most accurate)
  2. Length
  3. Angle/Slope
  4. Area
  5. Volume
  6. Color saturation/hue (least accurate)
- **Matching Data Types to Channels**:
  - Quantitative: Position, length, area
  - Ordinal: Position, color saturation
  - Categorical: Color hue, shape, position
- **Pre-attentive Processing**: Some visual features are perceived instantly (color, size, orientation)

### 3. Gestalt Principles and Perception
- **Key Principles**:
  - **Proximity**: Elements close together are perceived as groups
  - **Similarity**: Similar elements are perceived as related
  - **Continuity**: Elements arranged on a line/curve are related
  - **Closure**: Mind completes incomplete shapes
  - **Common Fate**: Elements moving together are grouped
  - **Figure-Ground**: Distinguishing foreground from background
- **Application**: Use these principles to guide viewer attention and create meaningful groupings

### 4. Color Theory and Color Blindness Accessibility
- **Color Models**: RGB (additive), CMYK (subtractive), HSL/HSV
- **Color Palettes**:
  - **Sequential**: For ordered data (light to dark)
  - **Diverging**: For data with meaningful midpoint
  - **Categorical/Qualitative**: For distinct categories
- **Color Blindness Considerations**:
  - ~8% of men, ~0.5% of women are color blind
  - Avoid red-green distinctions
  - Use colorblind-safe palettes (Viridis, ColorBrewer)
  - Add redundant encoding (patterns, labels)
- **Tools**: ColorBrewer, Coblis (simulator), Viz Palette

### 5. Typography and Layout
- **Typography Best Practices**:
  - Use sans-serif fonts for screens
  - Maintain consistent font hierarchy
  - Ensure readable font sizes (min 10-12pt for labels)
  - Limit font variations (max 2-3 fonts)
- **Layout Principles**:
  - Visual hierarchy guides eye movement
  - White space improves readability
  - Alignment creates order and professionalism
  - Grid systems ensure consistency
- **Title and Labels**: Clear, descriptive, with units

### 6. Information Density and Data-Ink Ratio
- **Edward Tufte's Principles**:
  - **Data-Ink Ratio** = Data Ink / Total Ink
  - Maximize data-ink ratio
  - Remove non-data ink (chartjunk)
  - Erase redundant data ink
- **Chartjunk Examples**:
  - 3D effects on 2D data
  - Unnecessary gridlines
  - Decorative elements
  - Excessive legends
- **Balance**: Too minimal can reduce comprehension; find optimal density

### 7. Grammar of Graphics
- **Conceptual Framework** (Leland Wilkinson, implemented in ggplot2):
  - **Data**: The dataset being visualized
  - **Aesthetics**: Mappings (x, y, color, size, shape)
  - **Geometry**: Visual elements (points, lines, bars)
  - **Facets**: Subplots by categories
  - **Statistics**: Transformations (binning, smoothing)
  - **Coordinates**: Coordinate system (Cartesian, polar)
  - **Themes**: Visual styling
- **Benefits**: Systematic approach to building visualizations layer by layer

### 8. Good vs Bad Visualization Examples
- **Bad Visualization Traits**:
  - Misleading scales (truncated y-axis)
  - 3D effects distorting perception
  - Poor color choices
  - Overloaded with information
  - Missing context/labels
- **Good Visualization Traits**:
  - Clear title and labels
  - Appropriate chart type
  - Honest representation
  - Accessible colors
  - Focused message
- **Common Mistakes**: Pie charts for many categories, dual y-axes, rainbow color maps

## Important Takeaways
1. Visual encoding should match data type to appropriate visual channel
2. Gestalt principles help guide viewer attention and perception
3. Always design for accessibility (color blindness, screen readers)
4. Maximize data-ink ratio while maintaining clarity
5. Grammar of Graphics provides a systematic framework for building visualizations
6. Good visualization tells a clear story without misleading the viewer
