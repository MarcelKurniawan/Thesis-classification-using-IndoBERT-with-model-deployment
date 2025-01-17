PGDMP  !                    |         	   thesis_db    16.1    16.1     �           0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                      false            �           0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                      false            �           0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                      false            �           1262    57344 	   thesis_db    DATABASE     �   CREATE DATABASE thesis_db WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_United States.1252';
    DROP DATABASE thesis_db;
                postgres    false                        2615    2200    public    SCHEMA        CREATE SCHEMA public;
    DROP SCHEMA public;
                pg_database_owner    false            �           0    0    SCHEMA public    COMMENT     6   COMMENT ON SCHEMA public IS 'standard public schema';
                   pg_database_owner    false    4            �            1259    57354    supervisor_data    TABLE     �   CREATE TABLE public.supervisor_data (
    supervisor_id character varying(5) NOT NULL,
    supervisor_name character varying(255) NOT NULL,
    slot integer
);
 #   DROP TABLE public.supervisor_data;
       public         heap    postgres    false    4            �            1259    57346    thesis_data    TABLE     �   CREATE TABLE public.thesis_data (
    id integer NOT NULL,
    title character varying(255) NOT NULL,
    abstract text,
    supervisor character varying(255),
    supervisor_id character varying(5),
    topic character varying(25)
);
    DROP TABLE public.thesis_data;
       public         heap    postgres    false    4            �            1259    57345    thesis_data_id_seq    SEQUENCE     �   CREATE SEQUENCE public.thesis_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 )   DROP SEQUENCE public.thesis_data_id_seq;
       public          postgres    false    216    4            �           0    0    thesis_data_id_seq    SEQUENCE OWNED BY     I   ALTER SEQUENCE public.thesis_data_id_seq OWNED BY public.thesis_data.id;
          public          postgres    false    215            T           2604    57349    thesis_data id    DEFAULT     p   ALTER TABLE ONLY public.thesis_data ALTER COLUMN id SET DEFAULT nextval('public.thesis_data_id_seq'::regclass);
 =   ALTER TABLE public.thesis_data ALTER COLUMN id DROP DEFAULT;
       public          postgres    false    216    215    216            �          0    57354    supervisor_data 
   TABLE DATA           O   COPY public.supervisor_data (supervisor_id, supervisor_name, slot) FROM stdin;
    public          postgres    false    217   ]       �          0    57346    thesis_data 
   TABLE DATA           \   COPY public.thesis_data (id, title, abstract, supervisor, supervisor_id, topic) FROM stdin;
    public          postgres    false    216   �       �           0    0    thesis_data_id_seq    SEQUENCE SET     A   SELECT pg_catalog.setval('public.thesis_data_id_seq', 1, false);
          public          postgres    false    215            X           2606    57365    supervisor_data pk_supervisor 
   CONSTRAINT     h   ALTER TABLE ONLY public.supervisor_data
    ADD CONSTRAINT pk_supervisor PRIMARY KEY (supervisor_name);
 G   ALTER TABLE ONLY public.supervisor_data DROP CONSTRAINT pk_supervisor;
       public            postgres    false    217            V           2606    57353    thesis_data thesis_data_pkey 
   CONSTRAINT     Z   ALTER TABLE ONLY public.thesis_data
    ADD CONSTRAINT thesis_data_pkey PRIMARY KEY (id);
 F   ALTER TABLE ONLY public.thesis_data DROP CONSTRAINT thesis_data_pkey;
       public            postgres    false    216            �   &  x�]�M��@���)� A���L0`�A��̦3c�vjO��S������^�z���aqgB��C�P�|��ss�T�m����.�B�j8���n�	��V�����TUضXB��A��!�h|=@u��B�񾟰�%3Z��P��Z��2b�(��������8�P�Pحٳ��f�Wz����m���O�\�d-��z!�u{�M��ض^y����;ޓ�S�����J3$Y'pY�z
��ה8v�!7��͓;tY*�K�	6d擗|�v�u����#�j��н'_�a��]�T      �      x������ � �     