/*******************************************************************************
 * rundags_mfd.h
 *
 * History:
 *    2019/09/11  - [Tao Wu] created
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.	   See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
******************************************************************************/

#ifndef _RUNDAGS_MFD_H_
#define _RUNDAGS_MFD_H_

int set_run_dags_struct_mfd(struct net_desc *pnet);

void update_run_dags_struct_port_mfd(struct net_desc *pnet);
void update_run_dags_struct_poke_mfd(struct net_desc *pnet);
int update_run_dags_struct_loop_cnt_mfd(struct net_desc *pnet,
	struct net_run_cfg *net_run);

int split_net_run_mfd(struct nnctrl_info *pctl, struct net_desc *pnet,
	int net_id, uint32_t split_num, float *vp_time_us);
int single_dag_run_mfd(struct nnctrl_info *pctl, struct net_desc *pnet,
	int net_id, float *vp_time_us);
int all_dag_run_mfd(struct nnctrl_info *pctl, struct net_desc *pnet,
	int net_id, float *vp_time_us);

#endif
