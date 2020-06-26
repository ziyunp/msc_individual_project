import folium as f

colors = ["red", "blue", "black", "green", "purple", "orange", "darkred", "darkblue", "darkgreen", "gray", "darkpurple"]

def make_map(map_df, cluster_field, save=False, map_file_name="", with_points=False):
  m = f.Map()
  m.fit_bounds([[map_df.Event_Lat.min(), map_df.Event_Long.min()],
                [map_df.Event_Lat.max(), map_df.Event_Long.max()]])

  i = 0

  for cluster in map_df[cluster_field].unique():
    df_temp_clust = map_df[map_df[cluster_field] == cluster]
    color = colors[i]
    if i < len(colors) - 1:
      i += 1
    else:
      i = 0
    fg = f.FeatureGroup(name=str(cluster), overlay=True, control=True)
    for leg_id in df_temp_clust["leg_id"].unique():
      df_temp = df_temp_clust[df_temp_clust['leg_id'] == leg_id]
      df_temp = df_temp.sort_values(by=['Event_Dttm'])
      fg.add_child(f.PolyLine(list(zip(df_temp.Event_Lat, df_temp.Event_Long)),
                              color=color, popup=str(leg_id)))
      if with_points:
        for lat, lon, leg_id in list(zip(df_temp.Event_Lat, df_temp.Event_Long, df_temp.leg_id)):
          f.Circle((lat, lon), color=color, radius=10).add_to(f.FeatureGroup(name=cluster).add_to(m))
    m.add_child(fg)
  
  f.LayerControl(collapsed=True).add_to(m)
  if save:
    print("Saving into {}".format(map_file_name))
    m.save(outfile=str(map_file_name) + ".html")

  f.LayerControl(collapsed=False).add_to(m)
  return m

# TODO: sort by event_dttm
def make_map_with_line_segments(lm, lines_N, distance_label, save=False, map_file_name="", with_points=False):
  m = f.Map()
  xm = [lm["p1"][0]] + [lm["p2"][0]]
  ym = [lm["p1"][1]] + [lm["p2"][1]]
  xn = [n["p1"][0] for n in lines_N] + [n["p2"][0] for n in lines_N]
  yn = [n["p1"][1] for n in lines_N] + [n["p2"][1] for n in lines_N]

  x_min = min(xm + xn)
  x_max = max(xm + xn)
  y_min = min(ym + yn)
  y_max = max(ym + yn)

  m.fit_bounds([[x_min, y_min], [x_max, y_max]])
  # plot lm
  fg = f.FeatureGroup(name="line_m", overlay=True, control=True)
  lm_coords = [lm["p1"]] + [lm["p2"]]
  fg.add_child(f.PolyLine(lm_coords, color=colors[0], popup=str(distance_label)))
  m.add_child(fg)
  if with_points:
    for lat, lon in lm_coords:
      f.Circle((lat, lon), color=colors[0], radius=10).add_to(f.FeatureGroup(name="line_m").add_to(m))

  # plot lines_N
  fg = f.FeatureGroup(name="lines_N", overlay=True, control=True)
  lines_N = sorted(lines_N, key=lambda x: x["dttm"])

  ln_coords = [n["p1"] for n in lines_N] + [n["p2"] for n in lines_N]
  fg.add_child(f.PolyLine(ln_coords, color=colors[1], popup="ln"))
  
  if with_points:
    for lat, lon in ln_coords:
      f.Circle((lat, lon), color=colors[1], radius=10).add_to(f.FeatureGroup(name="lines_N").add_to(m))

  m.add_child(fg)
  f.LayerControl(collapsed=True).add_to(m)

  if save:
    print("Saving into {}".format(map_file_name))
    m.save(outfile=str(map_file_name) + ".html")

  f.LayerControl(collapsed=False).add_to(m)
  return m
